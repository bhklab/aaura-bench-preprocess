# Changelog

## [0.2.0](https://github.com/bhklab/aaura-bench-preprocess/compare/v0.1.0...v0.2.0) (2026-09-01)


### Features

* add append index handling, not completely working yet, add some logging ([c02375a](https://github.com/bhklab/aaura-bench-preprocess/commit/c02375a335bb611ad3e8eeec2045a0f789fd0191))
* add bias correction as part of processing for MR ([78e2eb3](https://github.com/bhklab/aaura-bench-preprocess/commit/78e2eb315388d3e7f0c05a2330fee2a397db48e9))
* add centered bbox coordinates to the aaura index ([6324da7](https://github.com/bhklab/aaura-bench-preprocess/commit/6324da7dcaba4f2730e60fcd03cc8d3cfacfa66d))
* add disease site input for process to handle data dir organization including that, update docstring with all inputs ([b4274f2](https://github.com/bhklab/aaura-bench-preprocess/commit/b4274f22933775435c64888eda98df0fafadc6ee))
* add functionality for masks with multiple volumes ([9b671c8](https://github.com/bhklab/aaura-bench-preprocess/commit/9b671c801ef33e84b178a06148bea583e1085561))
* add handling for disease site in data directory structure ([0bf9364](https://github.com/bhklab/aaura-bench-preprocess/commit/0bf93644a6347d3e24f32f0fcc31614b796c52ac))
* add handling of masks with no labels ([9ae9cc8](https://github.com/bhklab/aaura-bench-preprocess/commit/9ae9cc8e7abcf262583eca35cbf1a8faec558aad))
* add logging for bias correction start and end ([3a17de1](https://github.com/bhklab/aaura-bench-preprocess/commit/3a17de1b92666d375f1d6f5ba8d18e7e520f8b2c))
* add mask_voxel_label to aaura index ([03daa24](https://github.com/bhklab/aaura-bench-preprocess/commit/03daa24eb639affad8a46e40dd369904498900d8))
* add multiple mask modality handling, dataset suffix and prefixes for handling OCSCC RADCURE ([d2da3d6](https://github.com/bhklab/aaura-bench-preprocess/commit/d2da3d607a4e322fcb53ad0da2b4febd51cc8a1f))
* add parallel processing ([98fa575](https://github.com/bhklab/aaura-bench-preprocess/commit/98fa57591d717cdea6810f1fa751dcba529857b3))
* added image modality validator to config ([a3dbe96](https://github.com/bhklab/aaura-bench-preprocess/commit/a3dbe96ca7ad0ffb8c75c935135c7d9d84203223))
* added pixi task for mit to aaura index creation ([e7728ee](https://github.com/bhklab/aaura-bench-preprocess/commit/e7728ee49605f3ccc1414804016f8ee7fbc82ac7))
* code used to develop nifti_to_aaura_index ([1139681](https://github.com/bhklab/aaura-bench-preprocess/commit/1139681c99e961162e0a1da6413e35a61efe38fb))
* combine all metadata functionality ([7b618af](https://github.com/bhklab/aaura-bench-preprocess/commit/7b618af315ecd2358db3f73b03f4930c357647cc))
* combine all metadata functionality ([b575d35](https://github.com/bhklab/aaura-bench-preprocess/commit/b575d35ed03ce5df4355081154ce36cf67f03380))
* combine all metadata functionality ([#5](https://github.com/bhklab/aaura-bench-preprocess/issues/5)) ([7b618af](https://github.com/bhklab/aaura-bench-preprocess/commit/7b618af315ecd2358db3f73b03f4930c357647cc))
* developed the rest of the main process pipeline ([467966d](https://github.com/bhklab/aaura-bench-preprocess/commit/467966d5a2fd8ef3814e74c9d28f4e39c0e91de5))
* handle metadata combination ([80a5d56](https://github.com/bhklab/aaura-bench-preprocess/commit/80a5d56fe32ef495958136e7139729c3a877f8b5))
* main data processing of nifti images for aaura benchmarking prep ([1588fff](https://github.com/bhklab/aaura-bench-preprocess/commit/1588fff0e8dd12a754b7796acb042daef213d3cc))
* make utility scripts for functions ([2290c2f](https://github.com/bhklab/aaura-bench-preprocess/commit/2290c2ffbd24180424d7250e27355c1201e56de3))
* more code for nifti_to_aaura_index ([2165da2](https://github.com/bhklab/aaura-bench-preprocess/commit/2165da25ec93332219f5d0cf97e65c33d8f084da))
* new general processing file for nifti datasets using pydantic ([21987ed](https://github.com/bhklab/aaura-bench-preprocess/commit/21987ed86650de017b4e4f0978956b04e92528a3))
* pydantic and n4 bias correction ([36f7d19](https://github.com/bhklab/aaura-bench-preprocess/commit/36f7d1980ce8dcc37a58f8fa988c8e0a2980dd99))
* pydantic and n4 bias correction ([36f7d19](https://github.com/bhklab/aaura-bench-preprocess/commit/36f7d1980ce8dcc37a58f8fa988c8e0a2980dd99))
* pydantic formula for AAuRA index row ([28e6c6f](https://github.com/bhklab/aaura-bench-preprocess/commit/28e6c6f6c734e429ab20881540e136b7d828e2c5))
* script to debug lesionlocator samples having no label 1 ([6600e19](https://github.com/bhklab/aaura-bench-preprocess/commit/6600e19ef5c32940875c5f582389bb8f97e3aea5))
* script to take mit_index and generate aaura_index ([bd78e91](https://github.com/bhklab/aaura-bench-preprocess/commit/bd78e91150b36d5027c1d4690ff6922004b1a654))
* start developing data preprocessing code for aaura benchmarking ([cfe91e1](https://github.com/bhklab/aaura-bench-preprocess/commit/cfe91e1c614d63c30395ea840ebcc2403b1cc481))


### Bug Fixes

* add logger setup ([158bd6c](https://github.com/bhklab/aaura-bench-preprocess/commit/158bd6c90bfa7cde7b225ee70fce12a6e7f90e75))
* add logging import ([0cc6a0c](https://github.com/bhklab/aaura-bench-preprocess/commit/0cc6a0ca4811b196859e6be4f3a8f910b4464f95))
* correct mask loading with updated mask_path setup ([6fccc2e](https://github.com/bhklab/aaura-bench-preprocess/commit/6fccc2ea97084f239f38719191417438fa9c3b2f))
