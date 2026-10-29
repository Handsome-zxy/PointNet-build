# Protocol freeze 2026-10-28

- protocol: `BuildingPointNet-v1-review-corrected`
- created: 2026-10-28T15:00:05.674389
- eval perm seed: 20261028
- checkpoint selection: validation mIoU at tuned threshold
- threshold selection: validation mIoU
- test strict unseen: False
- data manifest hash: `e9bbd0f26d72867085a2261ec73429cf9a8bf4112d8d2b2eba0c0944f907c0de`

## Code hashes

| file | sha256 |
| --- | --- |
| `DESIGN.md` | `dcf6f0f29395027ed5b27f2325a3c9f0c10d565e8f2ae834c14ebe1eaf26e6d7` |
| `README.md` | `3aa660a6cd8f629d9b3684d204f5535abb666ea1b165c6afe768a366bd355099` |
| `cache/manifest.json` | `e9bbd0f26d72867085a2261ec73429cf9a8bf4112d8d2b2eba0c0944f907c0de` |
| `configs/base.json` | `16fc62a6085eaff22d2a220f370a398e2023d087f6aa4568962665f2eaf89fc4` |
| `configs/tnet_off.json` | `41013107e11f43c8fa115e537b01b1b9f9fe801556f97806a40346bdbe291975` |
| `configs/tnet_on.json` | `d3f6ef75352a6e386e000c241285cac0db6010b9a18c12d5e7d3d71e406cc660` |
| `reports/test_set_usage.md` | `6e497545613f3f5a7da67d38eac2f5f043f05a13b809154316548ce6850c5c03` |
| `scripts/audit_eval_grouping.py` | `af7aa2a5ae7090806ee29e156bed232c4506d82a337fc1bf536f4b15bfe6849d` |
| `scripts/evaluate_final_test.py` | `7b066a21afa8fb8b1a1b73d5bd5d6da4652af589f77e092ef4710e0b55fbd77b` |
| `scripts/export_val_errors.py` | `80f1589e46521da1b8d3c040415b1d17d52ee4a4b6286ca62278426805a1fabf` |
| `scripts/freeze_final_test_candidate.py` | `718ebab4a2f24c292f6b334941b1eaa3a6cb5a38e231243db9d1d2be74a027a9` |
| `scripts/freeze_protocol.py` | `71952fe75771b2b879f7f6808490fb2e6a60c6feadf3211534308722b33353bb` |
| `scripts/prepare_data.py` | `3507c517f6d81c96164374a0498d32c39f662d2ac000adffe14951b5f3e6bb08` |
| `scripts/prepare_final_test_candidate.py` | `28ee9484b53a40d4dfc1b745a580a8329929560d5a9bfabea9ca28990db84005` |
| `scripts/reevaluate_checkpoint.py` | `5e00a5db967cfc5ecf3eeb455c60f39c0c175e1c7054fde721e5bfae27b89196` |
| `scripts/run_experiment.py` | `ea31ec26f1304e25fdab0272f3a35e58c0a14a14572262605b2505e7412797f8` |
| `src/building_pointnet/__init__.py` | `3f67367bdf6e941aaeb5158606fc19301b0153abb195211cf94b52f6f081c919` |
| `src/building_pointnet/config.py` | `9cd02f5892616eb2b3869d0c1ddaab3c585543c570a005aeb6f7da1736f34dfd` |
| `src/building_pointnet/data_prep.py` | `03ba84f1bf335e8cfa33a4645a3c26cf45f96e78b9d526fe8dd696fc8a45da53` |
| `src/building_pointnet/dataset.py` | `f211844c472348cd1cb28c6eec26445041889ffa93867cfd3743cc2a52456740` |
| `src/building_pointnet/evaluate.py` | `e60375dbe105a65fd92e5fd58be4148dbd7f19253298fef841f9f6a0dc9e5aa4` |
| `src/building_pointnet/las_utils.py` | `6efcf0b5ad29a98a948ec2f95c340b33333cef6b555bdd69a342d7766e1843be` |
| `src/building_pointnet/logging_utils.py` | `32d3d05c2d750a13e1d5a49a47d07ee3fffc599d5cf38da7442302b677cbf6c2` |
| `src/building_pointnet/metrics.py` | `8cf01f5d5213092a603a20100f834b770a52840c02a323b871e09a3355d8a214` |
| `src/building_pointnet/model.py` | `dbc79197339a542381aa4fc11c6c65f55c285b1b6ca7edf36b481eb3b84e4205` |
| `src/building_pointnet/predict.py` | `1a2225053b85530f7ef95a3a5102bc3a44164e022aecf99802fe16ca82d979f3` |
| `src/building_pointnet/tf_util.py` | `86d41300dc984c72652d0301cfeeb7796406856669ff37661220c8b93a36ad93` |
| `src/building_pointnet/train.py` | `dbca6ef110562ee7168935fbb3da70248fe47840c447c1e6b5735c0e44c34097` |
| `src/building_pointnet/transform_nets.py` | `a1adfb25b6611cf44a5dbd3d21605f03088f14d7a50191b5d7b0674df8254471` |
| `tests/test_metrics.py` | `52f0191631cc0e7eb48e502454af0e007c3161a36ddccade11c9edc878010b31` |
