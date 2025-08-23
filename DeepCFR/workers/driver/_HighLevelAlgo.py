import time

import ray

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.rl.base_cls.HighLevelAlgoBase import HighLevelAlgoBase as _HighLevelAlgoBase


class HighLevelAlgo(_HighLevelAlgoBase):

    def __init__(self, t_prof, la_handles, ps_handles, chief_handle):
        super().__init__(t_prof=t_prof, chief_handle=chief_handle, la_handles=la_handles)
        self._ps_handles = ps_handles
        self._all_p_aranged = list(range(self._t_prof.n_seats))

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        self._adv_args = t_prof.module_args["adv_training"]
        if self._AVRG:
            self._avrg_args = t_prof.module_args["avrg_training"]

        if self._t_prof.log_verbose:
            self._exp_adv_loss_handles = [
                self._ray.remote(self._chief_handle.create_experiment, f"P{p}/Adv_Loss")
                for p in range(self._t_prof.n_seats)
            ]
            self._exp_avrg_loss_handles = [
                self._ray.remote(self._chief_handle.create_experiment, f"P{p}/Avrg_Loss")
                for p in range(self._t_prof.n_seats)
            ] if self._AVRG else None
        else:
            self._exp_adv_loss_handles = None
            self._exp_avrg_loss_handles = None

    def _safe_wait(self, obj_refs):
        try:
            return self._ray.wait(obj_refs)
        except ray.exceptions.RayActorError as e:
            self._handle_actor_error(e)
            return [], []

    def _safe_get(self, obj_refs):
        try:
            return self._ray.get(obj_refs)
        except ray.exceptions.RayActorError as e:
            self._handle_actor_error(e)
            return []

    def _handle_actor_error(self, exc):
        actor_id = getattr(exc, "actor_id", None)
        if actor_id is None:
            return
        actor_hex = actor_id.hex() if hasattr(actor_id, "hex") else actor_id

        alive_las = [la for la in self._la_handles if la._actor_id.hex() != actor_hex]
        if len(alive_las) != len(self._la_handles):
            print(f"Learner actor {actor_hex} failed. Removing handle.")
            self._la_handles = alive_las
            try:
                self._ray.wait([
                    self._ray.remote(self._chief_handle.update_alive_las, self._la_handles)
                ])
            except ray.exceptions.RayActorError:
                print("Chief actor failed while updating alive learner actors.")

        for i, h in enumerate(self._ps_handles):
            if h is not None and h._actor_id.hex() == actor_hex:
                print(f"Parameter server {actor_hex} failed. Removing handle.")
                self._ps_handles[i] = None

    def init(self):
        # """"""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""
        if self._AVRG:
            self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged,
                                       update_avrg_for_plyrs=self._all_p_aranged)

        # """"""""""""""""""""""
        # NOT Deep CFR
        # """"""""""""""""""""""
        else:
            self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged)

    def run_one_iter_alternating_update(self, cfr_iter):
        t_generating_data = 0.0
        t_computation_adv = 0.0
        t_syncing_adv = 0.0

        for p_learning in range(self._t_prof.n_seats):
            self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged)
            print("Generating Data...")
            t0 = time.time()
            self._generate_traversals(p_id=p_learning, cfr_iter=cfr_iter)
            t_generating_data += time.time() - t0

            print("Training Advantage Net...")
            _t_computation_adv, _t_syncing_adv = self._train_adv(p_id=p_learning, cfr_iter=cfr_iter)
            t_computation_adv += _t_computation_adv
            t_syncing_adv += _t_syncing_adv

            if self._SINGLE:
                print("Pushing new net to chief...")
                self._push_newest_adv_net_to_chief(p_id=p_learning, cfr_iter=cfr_iter)

        print("Synchronizing...")
        self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged)

        return {
            "t_generating_data": t_generating_data,
            "t_computation_adv": t_computation_adv,
            "t_syncing_adv": t_syncing_adv,
        }

    def train_average_nets(self, cfr_iter):
        print("Training Average Nets...")
        t_computation_avrg = 0.0
        t_syncing_avrg = 0.0
        for p in range(self._t_prof.n_seats):
            _c, _s = self._train_avrg(p_id=p, cfr_iter=cfr_iter)
            t_computation_avrg += _c
            t_syncing_avrg += _s

        return {
            "t_computation_avrg": t_computation_avrg,
            "t_syncing_avrg": t_syncing_avrg,
        }

    def _train_adv(self, p_id, cfr_iter):
        t_computation = 0.0
        t_syncing = 0.0

        ps = self._ps_handles[p_id]
        if ps is None:
            return t_computation, t_syncing
        self._safe_wait([
            self._ray.remote(ps.reset_adv_net, cfr_iter)
        ])
        self._update_leaner_actors(update_adv_for_plyrs=[p_id])

        SMOOTHING = 200
        accumulated_averaged_loss = 0.0
        for epoch_nr in range(self._adv_args.n_batches_adv_training):
            t0 = time.time()
            global_step = (
                cfr_iter * self._adv_args.n_batches_adv_training + epoch_nr
            )

            # Compute gradients
            grads_from_all_las, _averaged_loss = self._get_adv_gradients(p_id=p_id)
            accumulated_averaged_loss += _averaged_loss

            t_computation += time.time() - t0

            # Applying gradients
            t0 = time.time()
            ps = self._ps_handles[p_id]
            if ps is None:
                return t_computation, t_syncing
            self._safe_wait([
                self._ray.remote(ps.apply_grads_adv,
                                 grads_from_all_las)
            ])

            # Step LR scheduler
            ps = self._ps_handles[p_id]
            if ps is None:
                return t_computation, t_syncing
            self._safe_wait([
                self._ray.remote(ps.step_scheduler_adv,
                                 _averaged_loss)
            ])

            # update ADV on all las
            self._update_leaner_actors(update_adv_for_plyrs=[p_id])

            # log current loss
            if self._t_prof.log_verbose and ((epoch_nr + 1) % SMOOTHING == 0):
                self._safe_wait([
                    self._ray.remote(
                        self._chief_handle.add_scalar,
                        self._exp_adv_loss_handles[p_id],
                        "DCFR_NN_Losses/Advantage",
                        global_step,
                        accumulated_averaged_loss / SMOOTHING,
                    )
                ])
                accumulated_averaged_loss = 0.0

            t_syncing += time.time() - t0

        return t_computation, t_syncing

    def _get_adv_gradients(self, p_id):
        grads_refs = [
            self._ray.remote(la.get_adv_grads,
                             p_id)
            for la in self._la_handles
        ]
        self._safe_wait(grads_refs)
        grads_vals = self._safe_get(grads_refs)
        grads_refs = [ref for ref, val in zip(grads_refs, grads_vals) if val is not None]

        losses = self._safe_get([
            self._ray.remote(la.get_loss_last_batch_adv,
                             p_id)
            for la in self._la_handles
        ])

        losses = [loss for loss in losses if loss is not None]

        n = len(losses)
        averaged_loss = sum(losses) / float(n) if n > 0 else -1

        return grads_refs, averaged_loss

    def _generate_traversals(self, p_id, cfr_iter):
        self._safe_wait([
            self._ray.remote(la.generate_data,
                             p_id, cfr_iter)
            for la in self._la_handles
        ])

    def _update_leaner_actors(self, update_adv_for_plyrs=None, update_avrg_for_plyrs=None):
        """

        Args:
            update_adv_for_plyrs (list):         list of player_ids to update adv for
            update_avrg_for_plyrs (list):        list of player_ids to update avrg for
        """

        assert isinstance(update_adv_for_plyrs, list) or update_adv_for_plyrs is None
        assert isinstance(update_avrg_for_plyrs, list) or update_avrg_for_plyrs is None

        _update_adv_per_p = [
            True if (update_adv_for_plyrs is not None) and (p in update_adv_for_plyrs) else False
            for p in range(self._t_prof.n_seats)
        ]

        _update_avrg_per_p = [
            True if (update_avrg_for_plyrs is not None) and (p in update_avrg_for_plyrs) else False
            for p in range(self._t_prof.n_seats)
        ]

        la_batches = []
        n = len(self._la_handles)
        c = 0
        while n > c:
            s = min(n, c + self._t_prof.max_n_las_sync_simultaneously)
            la_batches.append(self._la_handles[c:s])
            if type(la_batches[-1]) is not list:
                la_batches[-1] = [la_batches[-1]]
            c = s

        w_adv = [None for _ in range(self._t_prof.n_seats)]
        w_avrg = [None for _ in range(self._t_prof.n_seats)]
        for p_id in range(self._t_prof.n_seats):
            ps = self._ps_handles[p_id]
            w_adv[p_id] = None if (not _update_adv_per_p[p_id] or ps is None) else self._ray.remote(
                ps.get_adv_weights)

            ps = self._ps_handles[p_id]
            w_avrg[p_id] = None if (not _update_avrg_per_p[p_id] or ps is None) else self._ray.remote(
                ps.get_avrg_weights)

        for batch in la_batches:
            self._safe_wait([
                self._ray.remote(la.update,
                                 w_adv,
                                 w_avrg)
                for la in batch
            ])

    # ____________ SINGLE only
    def _push_newest_adv_net_to_chief(self, p_id, cfr_iter):
        ps = self._ps_handles[p_id]
        if ps is None:
            return
        self._safe_wait([self._ray.remote(self._chief_handle.add_new_iteration_strategy_model,
                                          p_id,
                                          self._ray.remote(ps.get_adv_weights),
                                          cfr_iter)])

    # ____________ AVRG only
    def _get_avrg_gradients(self, p_id):
        grads_refs = [
            self._ray.remote(la.get_avrg_grads,
                             p_id)
            for la in self._la_handles
        ]
        self._safe_wait(grads_refs)
        grads_vals = self._safe_get(grads_refs)
        grads_refs = [ref for ref, val in zip(grads_refs, grads_vals) if val is not None]

        losses = self._safe_get([
            self._ray.remote(la.get_loss_last_batch_avrg,
                             p_id)
            for la in self._la_handles
        ])

        losses = [loss for loss in losses if loss is not None]

        n = len(losses)
        averaged_loss = sum(losses) / float(n) if n > 0 else -1

        return grads_refs, averaged_loss

    def _train_avrg(self, p_id, cfr_iter):
        t_computation = 0.0
        t_syncing = 0.0

        ps = self._ps_handles[p_id]
        if ps is None:
            return t_computation, t_syncing
        self._safe_wait([self._ray.remote(ps.reset_avrg_net)])
        self._update_leaner_actors(update_avrg_for_plyrs=[p_id])

        SMOOTHING = 200
        accumulated_averaged_loss = 0.0

        if cfr_iter > 0:
            for epoch_nr in range(self._avrg_args.n_batches_avrg_training):
                t0 = time.time()
                global_step = (
                    cfr_iter * self._adv_args.n_batches_adv_training + epoch_nr
                )

                # Compute gradients
                grads_from_all_las, _averaged_loss = self._get_avrg_gradients(p_id=p_id)
                accumulated_averaged_loss += _averaged_loss

                t_computation += time.time() - t0

                # Applying gradients
                t0 = time.time()
                ps = self._ps_handles[p_id]
                if ps is None:
                    return t_computation, t_syncing
                self._safe_wait([
                    self._ray.remote(ps.apply_grads_avrg,
                                     grads_from_all_las)
                ])

                # Step LR scheduler
                ps = self._ps_handles[p_id]
                if ps is None:
                    return t_computation, t_syncing
                self._safe_wait([
                    self._ray.remote(ps.step_scheduler_avrg,
                                     _averaged_loss)
                ])

                # update AvrgStrategyNet on all las
                self._update_leaner_actors(update_avrg_for_plyrs=[p_id])

                # log current loss
                if self._t_prof.log_verbose and ((epoch_nr + 1) % SMOOTHING == 0):
                    self._safe_wait([
                        self._ray.remote(
                            self._chief_handle.add_scalar,
                            self._exp_avrg_loss_handles[p_id],
                            "DCFR_NN_Losses/Average",
                            global_step,
                            accumulated_averaged_loss / SMOOTHING,
                        )
                    ])
                    accumulated_averaged_loss = 0.0

                t_syncing += time.time() - t0

        return t_computation, t_syncing
