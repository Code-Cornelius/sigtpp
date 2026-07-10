import os
import unittest
from unittest.mock import patch

from config import ROOT_DIR
from test.paper_experiments.sig_degree_selection import _extract_sig_degree
from test.paper_experiments.training_helpers import (
    get_experiment_entry,
    get_model_name,
    load_experiment_config,
    parse_model_dir_to_cfg,
)


class TestLoadExperimentConfig(unittest.TestCase):

    def test_two_part_path_merges_experiment_yaml_into_shared_model_config(self):
        cfg = load_experiment_config("poisson_three_marks/ddpm_test.yaml")

        self.assertEqual(cfg["experiment_type"], "poisson_three_marks")
        self.assertEqual(cfg["version"], "ddpm")
        self.assertEqual(cfg["num_marks"], 3)
        self.assertEqual(cfg["mark_probs"], [0.7, 0.05, 0.25])

    def test_experiment_yaml_supplies_dataset_specific_params(self):
        cfg = load_experiment_config("hawkes_3x3/ddpm_test.yaml")

        self.assertEqual(cfg["experiment_type"], "hawkes_3x3")
        self.assertEqual(cfg["version"], "ddpm")
        self.assertEqual(cfg["time_max"], 15.0)
        self.assertEqual(cfg["baseline"], [0.5, 0.5, 0.5])
        self.assertEqual(cfg["adjacency"], [[0.5, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]])
        self.assertEqual(cfg["decays"], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_model_yaml_wins_on_conflict(self):
        """Model YAML values must overwrite experiment.yaml values on conflict."""
        # Patch _load_yaml_file so both calls return controlled data with a shared key (seeds).
        # hawkes_3x3/experiment.yaml exists on disk, so the merge branch is taken.
        with patch("test.paper_experiments.training_helpers._load_yaml_file") as mock_load:
            mock_load.side_effect = [
                {"version": "ddpm", "seeds": [42], "n_bootstraps": 5},  # model YAML
                {"seeds": [999], "time_max": 15.0},  # experiment.yaml with conflicting seeds
            ]
            cfg = load_experiment_config("hawkes_3x3/ddpm_test.yaml")

        self.assertEqual(cfg["seeds"], [42])  # model YAML wins
        self.assertEqual(cfg["time_max"], 15.0)  # experiment.yaml fills in

    def test_experiment_type_is_injected_from_path_not_experiment_yaml(self):
        # poisson_three_marks/experiment.yaml has no experiment_type key,
        # but even if it did, the path-injected value must win.
        cfg = load_experiment_config("poisson_three_marks/ddpm_test.yaml")
        self.assertEqual(cfg["experiment_type"], "poisson_three_marks")

    def test_no_experiment_yaml_still_loads(self):
        # hawkes/experiment.yaml declares no num_marks/mark_probs, so those keys
        # must not be injected into the merged config.
        cfg = load_experiment_config("hawkes/ddpm_test.yaml")
        self.assertEqual(cfg["experiment_type"], "hawkes")
        self.assertEqual(cfg["version"], "ddpm")
        self.assertNotIn("num_marks", cfg)
        self.assertNotIn("mark_probs", cfg)

    def test_sigtpp_config_uses_relative_sig_degree_grid(self):
        cfg = load_experiment_config("poisson_three_marks/sigtpp_test.yaml")

        params = cfg["parameter_sets"]
        self.assertEqual(params["relative_sig_degree"], [0])
        self.assertNotIn("use_degree_detector", params)
        self.assertNotIn("sig_degree", params)

    def test_sigtpp_run_name_grammar_roundtrips(self):
        """Pin the run-name grammar contracts the sig-degree ablation relies on.

        Names must be generated from the REAL shared sigtpp grid, not invented:
        the ablation's degree extraction and recompute_bootstrap's inverse cfg
        parser both consume these names, and hand-written fixtures previously
        masked that neither worked for the actual grid. The grid toggles between
        two mutually-exclusive sig-degree modes (`relative_sig_degree` vs the
        absolute `sig_degree`), so this test discovers whichever mode is live
        rather than assuming one, keeping it stable across paper-param sweeps.
        """
        cfg = load_experiment_config("poisson_three_marks/sigtpp.yaml")
        params = {key: values[0] for key, values in cfg["parameter_sets"].items()}
        sig_keys = {"relative_sig_degree", "sig_degree"} & params.keys()
        self.assertEqual(len(sig_keys), 1, f"expected exactly one sig-degree mode key, got {sig_keys}")
        sig_key = sig_keys.pop()
        sig_value = -2 if sig_key == "relative_sig_degree" else 2
        self.assertIn("use_teacher_forcing", params)
        self.assertIn("use_lr_scheduler", params)
        params[sig_key] = sig_value
        params["use_teacher_forcing"] = False
        params["use_lr_scheduler"] = True

        name = get_model_name("taxi", "sigtpp", None, params, 1000.0)

        # Contract 1: the ablation's degree extraction understands real names.
        self.assertEqual(_extract_sig_degree(name), sig_value)

        # Contract 2: the inverse parser (recompute_bootstrap's cfg resolution)
        # recovers the grid values, distinguishing the two use_* booleans.
        configs_root = os.path.join(ROOT_DIR, "test", "paper_experiments", "configs")
        parsed_name = get_model_name("x", "sigtpp", None, params, 1000.0)
        out_cfg = parse_model_dir_to_cfg(
            parsed_name, "poisson_three_marks", configs_root, ref_cfg_override=cfg
        )
        recovered = out_cfg["parameter_sets"]
        self.assertEqual(recovered[sig_key], sig_value)
        self.assertIs(recovered["use_teacher_forcing"], False)
        self.assertIs(recovered["use_lr_scheduler"], True)
        self.assertEqual(recovered["hid_size_rep"], params["hid_size_rep"])
        self.assertEqual(recovered["lr_gen"], params["lr_gen"])

    def test_parse_model_dir_rejects_legacy_ambiguous_use_tokens(self):
        """Legacy sigtpp dirs carry two `use_<bool>` tokens; parsing must fail
        loudly (duplicate key) instead of silently overwriting one boolean with
        the other. Only the colliding tokens are needed here; other grid keys
        (hid_size_rep, mark_loss_weight, ...) are irrelevant to this contract
        and would otherwise couple this fixture to today's specific grid."""
        cfg = load_experiment_config("poisson_three_marks/sigtpp.yaml")
        configs_root = os.path.join(ROOT_DIR, "test", "paper_experiments", "configs")
        legacy_name = "x_sigtpp_TX1000_use_F_use_T"
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            parse_model_dir_to_cfg(legacy_name, "poisson_three_marks", configs_root, ref_cfg_override=cfg)

    def test_get_experiment_entry_returns_registered_factories(self):
        entry = {"data_factory": object()}
        self.assertIs(
            get_experiment_entry("earthquake", registry={"earthquake": entry}), entry
        )

    def test_get_experiment_entry_raises_readable_error_with_suggestion(self):
        registry = {"earthquake": {}, "yelp_mississauga": {}}

        with self.assertRaisesRegex(
            ValueError,
            r"Unknown experiment_type 'yelp'.*Available experiments: earthquake, yelp_mississauga\..*Did you mean 'yelp_mississauga'\?",
        ):
            get_experiment_entry("yelp", registry=registry)
