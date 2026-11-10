import copy
import io
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from foolproof.architecture import validate
from foolproof.model import GPT, ModelConfig, Block, PartialBlock

torch.set_num_threads(1)
SPEC = dict(version=1,width=64,heads=2,context=32,blocks=[
    {"type":"attention"},{"type":"feed_forward"},{"type":"transformer"}])

class BuilderTests(unittest.TestCase):
    def test_order_training_checkpoint_generation(self):
        config, count = validate(SPEC, 256)
        model = GPT(ModelConfig(vocab_size=256, **config))
        self.assertEqual(model.num_params(), count)
        self.assertEqual([getattr(b, "kind", "transformer") for b in model.blocks],
                         ["attention","feed_forward","transformer"])
        x=torch.randint(0,256,(2,16))
        _, loss=model(x,x); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        model.eval()
        expected=model(x)[0].detach()
        f=io.BytesIO(); torch.save(dict(config=model.cfg.to_dict(),state=model.state_dict()),f); f.seek(0)
        saved=torch.load(f,weights_only=True)
        restored=GPT(ModelConfig.from_dict(saved["config"])); restored.load_state_dict(saved["state"]); restored.eval()
        torch.testing.assert_close(expected, restored(x)[0])
        self.assertEqual(len(list(restored.generate(x[:1,:4],3))),3)
        for key in ("n_layer","n_head","n_embd","block_size","block_types"):
            self.assertEqual(saved["config"][key],config[key])

    def test_invalid(self):
        invalid=[{},None,{"version":2},dict(SPEC,width=True),dict(SPEC,width=65),
                 dict(SPEC,blocks=[]),dict(SPEC,blocks=[{"type":"feed_forward"}]),
                 dict(SPEC,blocks=[{"type":"evil"}]),dict(SPEC,heads=0),
                 dict(SPEC,context=100000),dict(SPEC,blocks=SPEC["blocks"]*6),
                 dict(SPEC,extra="bad"),dict(SPEC,blocks=[{"type":[]}])]
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate(spec,256)

    def test_run_validation_and_persistence(self):
        from foolproof import runs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); models=root/"models"; models.mkdir()
            data=root/"data"; (data/"sample").mkdir(parents=True)
            (data/"sample"/"tokenizer.json").write_text("{}")
            ds=dict(slug="sample",vocab_size=256,n_train=1000)
            with patch("foolproof.paths.models_dir",return_value=models), patch("foolproof.paths.datasets_dir",return_value=data):
                with self.assertRaises(ValueError):
                    runs.create("bad",ds,"pocket",1,{},architecture=dict(SPEC,heads=0))
                self.assertEqual(list(models.iterdir()),[])
                job=runs.create("good",ds,"pocket",1,{},architecture=SPEC)
                self.assertEqual(job["model"]["block_types"],["attention","feed_forward","transformer"])
                import json
                cfg=json.loads((Path(job["run_dir"])/"config.json").read_text())
                self.assertEqual(cfg["model"],job["model"])
                self.assertEqual(runs.job_for(cfg,{},True)["model"],job["model"])

    def test_legacy(self):
        model=GPT(ModelConfig(vocab_size=256,n_layer=2,n_head=2,n_embd=64,block_size=32))
        self.assertTrue(all(isinstance(b,Block) for b in model.blocks))
        self.assertEqual(model.cfg.block_types,None)

    def test_reorder(self):
        spec=copy.deepcopy(SPEC); spec["blocks"].reverse()
        cfg,_=validate(spec,256); model=GPT(ModelConfig(**cfg))
        self.assertEqual([getattr(b,"kind","transformer") for b in model.blocks],["transformer","feed_forward","attention"])

if __name__=="__main__": unittest.main()
