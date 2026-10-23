<h1 align="center">OPTIC</h1>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<p align="center">
  <strong>Extreme Compression of Per-Layer Embeddings<br>via Cross-Layer Latent Vector Quantization</strong>
</p>

<p align="center">
  Jiacheng Liu<sup>1</sup> · Yaxin Luo<sup>1</sup> · Xiaohan Zhao<sup>1</sup> · Peng Sun<sup>2,3</sup> · Xinyi Shang<sup>1,4</sup> · Jiacheng Cui<sup>1</sup> · Yonina C. Eldar<sup>5</sup> · Zhiqiang Shen<sup>1*</sup>
</p>

<p align="center">
  <sup>1</sup>Mohamed bin Zayed University of Artificial Intelligence · <sup>2</sup>Westlake University · <sup>3</sup>Zhejiang University<br>
  <sup>4</sup>University College London · <sup>5</sup>Weizmann Institute of Science<br>
  <sup>*</sup>通讯作者
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.31309"><img src="https://img.shields.io/badge/arXiv-2610.31309-b31b1b?style=flat-square" alt="arXiv 论文"></a>
  <a href="https://huggingface.co/zcahjl3/gemma-4-E4B-it-OPTIC-100x"><img src="https://img.shields.io/badge/Hugging_Face-Model-ffd21e?style=flat-square" alt="Hugging Face 模型"></a>
  <a href="#method-video"><img src="https://img.shields.io/badge/Video-中文-58a6ff?style=flat-square" alt="中文方法视频"></a>
</p>

Gemma 4 E2B 和 E4B 为每个词元在每个解码层保存独立的逐层嵌入（Per-Layer
Embedding，PLE）。在 E4B 中，PLE 占模型参数的 35.2%，以 BF16 存储时需要
5.64 GB。OPTIC（Output-Path-fitted, Token-shared Indexed Codes）为每个词元
保存一组跨层共享的量化索引，再用各层的仿射解码器重建对应的嵌入向量。
码本和解码器通过最小化 PLE 输出投影处的误差来拟合。

[![每个词元的一组共享编码供所有 PLE 层使用。](docs/figures/shared-codes.png)](docs/figures/shared-codes.pdf)

<p align="center"><em>图 1. OPTIC 跨层共享词元编码，并在输出投影后拟合重建的 PLE 分支贡献。</em></p>

<a name="method-video"></a>
<details open>
<summary>观看方法讲解 · 3:28</summary>

https://github.com/user-attachments/assets/ef158c17-e889-428f-b76c-8acfc6d08d55

</details>

## 目录

- [主要结果](#主要结果)
- [快速开始](#快速开始)
- [论文结果](#论文结果)
- [基准评测](#基准评测)
- [拟合 OPTIC](#拟合-optic)
- [代码结构](#代码结构)
- [致谢](#致谢)
- [引用](#引用)
- [许可证](#许可证)

## 主要结果

- 在约 100× PLE 压缩率下，OPTIC 保留 BF16 Gemma 4 E4B 在七项基准上
  97.8% 的综合性能。六种常规基线中最强的乘积量化（PQ）保留 66.3%。
- 对官方 W4A16 E4B 版本，同等压缩率下的性能保留率为 96.9%。
- 对 BF16 E4B，594× 和 1002× 压缩下分别保留 90.5% 和 84.1% 的性能。
  在 1002× 压缩率下，E4B 的 PLE 仅占 5.6 MB。
- 在单张 RTX 4090、固定 1 GiB KV cache 的设置下，以 100× 编码提供推理服务，
  BF16 和 W4A16 E4B 的 GPU 显存用量分别降低 29.7% 和 43.1%；
  实际运行时间最多分别增加 0.40% 和 0.69%。

## 快速开始

使用 Python 3.12 和支持 CUDA 的 GPU。24 GiB 显存可以运行采用 OPTIC 的
BF16 Gemma 4 E4B，并完成下面的短文本生成示例。在 Linux shell 中运行：

```bash
git clone https://github.com/VILA-Lab/optic.git
cd optic
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m pip install -e .
python examples/generate.py --prompt "Explain why the sky is blue in two sentences."
```

首次运行会下载固定版本的模型和
[100× OPTIC 压缩产物](https://huggingface.co/zcahjl3/gemma-4-E4B-it-OPTIC-100x)，
后续运行复用缓存。回答输出到标准输出，耗时和 GPU 显存用量输出到标准错误。

图片问答使用本地图片文件：

```bash
python examples/generate.py --image image.png --prompt "Describe this image."
```

也可以用一个小型合成表查看码本查表和解码过程，无需加载模型或使用 GPU：

```bash
python examples/decode_optic.py
```

输出形状为 `[3, 3, 16]`，分别对应三个请求的词元行、三层和每层 16 个值。
核心方法位于 [builders/methods/optic.py](builders/methods/optic.py)，
紧凑解码器位于 [compact_adapter.py](src/ple_eval/compact_adapter.py)。

模型加载时跳过原始的 5.64 GB PLE 张量，只在 GPU 上保留 56.4 MB 的 OPTIC
张量。推理时，仅对请求的词元行进行解码，通过一次堆叠矩阵乘法重建各层嵌入。
本地模型路径、vLLM 推理和 OpenAI 兼容服务的用法见[部署指南](docs/deployment.md)。

## 论文结果

名义 100× 压缩档位的结果如下。每个单元格给出性能保留率（百分比），
括号内为实际 PLE 压缩率。

| 方法 | E2B BF16 | E4B BF16 | E2B W4A16 | E4B W4A16 |
| --- | ---: | ---: | ---: | ---: |
| *基线* | | | | |
| PQ | 22.9 (101.24×) | 66.3 (101.43×) | 0.0 (101.24×) | 82.4 (101.43×) |
| RVQ | 0.0 (100.96×) | 37.6 (98.64×) | 29.3 (100.96×) | 82.0 (101.20×) |
| OPQ-P | 0.0 (100.92×) | 52.4 (101.16×) | 0.0 (100.92×) | 57.9 (101.16×) |
| AQ | 0.0 (100.96×) | 45.6 (101.20×) | 0.0 (100.96×) | 62.8 (101.20×) |
| AQLM Euclidean beam encoder | 20.7 (100.96×) | 46.7 (101.20×) | 32.0 (100.96×) | 82.8 (101.20×) |
| INT8 randomized low-rank SVD | 0.0 (100.13×) | 37.7 (99.76×) | 0.0 (100.13×) | 51.1 (100.24×) |
| *我们的方法* | | | | |
| Output-path-weighted PCA-PQ | 32.2 (96.14×) | 93.3 (96.36×) | 44.5 (96.14×) | 88.3 (96.36×) |
| OPTIC | 89.1 (99.86×) | **97.8** (99.94×) | 91.5 (99.86×) | **96.9** (99.94×) |
| Frequency-tiered OPTIC | **90.0** (100.48×) | 95.7 (100.00×) | **93.0** (100.48×) | 95.1 (100.00×) |

性能保留率将每项任务的分数与使用未压缩 PLE 的同一 checkpoint 比较。
各任务分数比值以 1 为上限，再对七项任务取等权几何平均：PIQA、GSM8K、
GPQA Diamond、AIME 2026、MMLU-Redux、IFBench 和 InfographicVQA。

```math
\mathrm{Retention} = 100\left[\prod_{k=1}^{7}\min\left(\frac{s_{m,k}}{s_{\mathrm{ref},k}},1\right)\right]^{1/7}
```

其中 $s_{m,k}$ 是方法 $m$ 在任务 $k$ 上的分数，$s_{\mathrm{ref},k}$ 是未压缩模型
的分数。所有任务的分数均未下降时，保留率为 100%；任一任务分数为零时，
保留率为零。压缩率是原始 BF16 PLE 的大小除以压缩后 PLE 的常驻内存字节数。
表中结果使用拟合种子 0，各任务分数见 [main-table.json](docs/figures/main-table.json)。

[![BF16 Gemma 4 E4B 上的压缩率与性能保留率曲线。](docs/figures/rate-retention.png)](docs/figures/rate-retention.pdf)

<p align="center"><em>图 2. BF16 Gemma 4 E4B 上的 PLE 压缩率与七项基准综合性能保留率。</em></p>

在 BF16 E4B 上，PQ 和 OPQ-P 基线（虚线）的性能保留率在接近 500× 和
1000× 压缩时降为零。

## 基准评测

[已发布的压缩产物](https://huggingface.co/zcahjl3/gemma-4-E4B-it-OPTIC-100x)
是 BF16 Gemma 4 E4B 的种子 0 OPTIC 拟合结果，PLE 压缩率为 99.94×，
对应上表中 97.8% 的性能保留率。
按[评测指南](docs/evaluation.md)安装评测环境后，运行七项基准：

```bash
ple-eval evaluate --gpus 0 --run-id optic-e4b
```

该命令加载固定版本的 checkpoint 和压缩产物，准备 PLE 重建，运行基准，
并将各任务分数及性能保留率写入 `results/optic-e4b/result.json`。
未压缩模型的参考分数已包含在 [main-table.json](docs/figures/main-table.json) 中。

通过 `--model-dir /path/to/snapshot` 复用本地 checkpoint，
通过 `--artifact /path/to/compressed_ple.npz` 复用本地压缩产物。
添加 `--smoke --run-id optic-smoke` 可先运行小规模 GPU 文本与图片检查。

## 拟合 OPTIC

[拟合指南](docs/fitting.md)包含校准数据准备，以及按已发布压缩产物的设置
对 BF16 Gemma 4 E4B 拟合 OPTIC 的步骤。
表中其他方法的设置见[方法配置](docs/baselines.md)。

## 代码结构

| 目录 | 内容 |
| --- | --- |
| [builders/methods](builders/methods) | OPTIC 和输出路径加权 PCA-PQ 的拟合实现。 |
| [builders/baselines](builders/baselines) | PQ、RVQ、PCA-PQ 和低秩 SVD 实现。 |
| [src/ple_eval](src/ple_eval) | 压缩嵌入解码器、vLLM 集成与基准评测。 |
| [scripts](scripts) | 校准、拟合和结果汇总命令。 |
| [configs](configs) | 模型、校准、拟合、评测和论文实验配置。 |
| [data](data) | 固定的校准输入和基准请求集。 |
| [examples](examples) | Transformers 与 vLLM 生成示例，以及 CPU 编码表解码。 |
| [docs](docs) | 推理、评测、拟合指南，方法配置和图表。 |

核心方法位于 [builders/methods/optic.py](builders/methods/optic.py)，
解码入口为 [`load_compact_ple`](src/ple_eval/compact_adapter.py)。

## 致谢

本工作得到 MBZUAI-WIS Joint Program for Artificial Intelligence Research 的支持。

评测使用了 [vLLM](https://github.com/vllm-project/vllm)、
[EvalScope](https://github.com/modelscope/evalscope) 和官方
[IFBench 评分器](https://github.com/allenai/IFBench)。AQLM 基线使用
[作者的 beam-search 实现](https://github.com/Vahe1994/AQLM)。
仓库中第三方代码与资源的来源及许可证见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 引用

使用 OPTIC 时请引用：

```bibtex
@misc{liu2026extremecompressionperlayerembeddings,
      title={Extreme Compression of Per-Layer Embeddings via Cross-Layer Latent Vector Quantization},
      author={Jiacheng Liu and Yaxin Luo and Xiaohan Zhao and Peng Sun and Xinyi Shang and Jiacheng Cui and Yonina C. Eldar and Zhiqiang Shen},
      year={2026},
      eprint={2610.31309},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.31309},
}
```

## 许可证

OPTIC 代码采用 [Apache License 2.0](LICENSE)。
