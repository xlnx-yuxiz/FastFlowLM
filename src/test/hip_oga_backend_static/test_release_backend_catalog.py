"""Release-facing backend package identities in the bundled FLM catalog."""

import json
from pathlib import Path
import unittest


CATALOG = Path(__file__).resolve().parents[2] / "model_list.json"


class ReleaseBackendCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = json.loads(CATALOG.read_text(encoding="utf-8"))["models"]

    def test_gemma_tag_keeps_npu_package_and_selects_public_hip_package(self):
        model = self.models["gemma3"]["4b"]
        self.assertEqual(model["name"], "Gemma3-4B-NPU2")
        self.assertIn("model.q4nx", model["files"])
        hip = model["hip_oga"]
        self.assertFalse(hip.get("hip_only", False))
        self.assertNotIn("unsupported_reason", hip)
        self.assertEqual(hip["offload_policy"], "strict_gpu")
        self.assertEqual(hip["config_path"], "hip-gemma3-4b-oga")
        self.assertEqual(hip["public_hf"]["repo_id"],
                         "amd/Gemma3-4B-it_gpuep_1.0_gpu_128")
        self.assertEqual(hip["public_hf"]["revision"],
                         "67e57aeb12f71ec7ac94068c433d8de075653863")

    def test_legacy_hip_tag_is_same_package_alias(self):
        canonical = self.models["gemma3"]["4b"]["hip_oga"]
        legacy = self.models["hip-gemma3"]["4b"]["hip_oga"]
        self.assertEqual(legacy["legacy_alias_for"], "gemma3:4b")
        self.assertTrue(legacy["hip_only"])
        self.assertEqual(legacy["config_path"], canonical["config_path"])
        self.assertEqual(legacy["public_hf"], canonical["public_hf"])
        self.assertNotIn("unsupported_reason", legacy)

    def test_unqualified_public_candidates_remain_gated(self):
        for family, size in (("hip-qwen2.5-coder", "14b"),
                             ("hip-qwen2.5", "14b"),
                             ("hip-qwen3.5-vl", "9b"),
                             ("hip-qwen3.6-vl", "35b-a3b")):
            with self.subTest(family=family):
                hip = self.models[family][size]["hip_oga"]
                self.assertTrue(hip["hip_only"])
                self.assertTrue(hip["unsupported_reason"])


if __name__ == "__main__":
    unittest.main()
