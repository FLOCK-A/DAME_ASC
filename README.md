# DAME-ASC — 设备条件化声学场景分类（占位实现）

本仓库包含 DAME-ASC 项目的启动骨架（占位实现），用于快速验证架构（多 expert + DCDIR 增强 + DCF 融合 + two-stage training/inference 流程）。代码以最小依赖实现接口与工具，便于逐步替换为真实模型与训练代码。

核心目标
- 提供可运行的端到端占位实现（数据加载、增强占位、多个 expert、device-conditioned fusion 占位、训练/推理脚本、分析工具）。
- 让你可以：快速做“有效性自检”（per-expert per-device 表）、验证 DCDIR 与 DCF 的潜在收益，再逐步引入真实模型（PaSST、BEATs、CNN 等）。

主要特性（占位）
- JSON manifest 支持（dict-list 与 list-of-lists/list4）
- Mel-EQ Bank（DCDIR Option A）的占位实现
- 两个占位 expert（`passt`、`cnn`）输出 logits，用于演示融合流程
- 占位 DCF 融合器（设备 embedding + gating + 可选 temperature）
- Stage-1 / Stage-2 占位训练脚本（示例损失：CE、consistency、reg）
- 推理脚本支持 per-device 回退逻辑与 TTA（占位）
- 工具：构建 per-expert per-device 表（`build_device_table`）

快速开始（Windows cmd）
1) 创建并激活虚拟环境，然后安装依赖：

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

说明：`requirements.txt` 含最小运行依赖（numpy、pytest、PyYAML 可选）。如果你要用 YAML 配置，请确保安装 PyYAML：

```cmd
pip install PyYAML
```

2) 运行占位推理（示例）：

```cmd
python main.py --config config.json --mode infer
```

3) 运行占位训练（Stage-1，smoke run）：

```cmd
python src/train/train_stage1.py --config config.json --workdir runs/stage1_test --epochs 1
```

4) 构建 per-device 精度表（工具）：

```cmd
python src/tools/build_device_table.py --val_manifest data/manifests/train.json --out_csv runs/per_device_table.csv
```

5) 推理（示例，支持 per-device 回退）：

```cmd
python src/infer/infer.py --config config.json --test_manifest data/manifests/train.json --general_ckpt runs/stage1_test/best.ckpt --device_ckpt_dir runs/stage2_test --out_csv runs/preds.csv
```

6) 运行单元测试：

```cmd
pytest -q
```

目录结构（简要）
- main.py — top-level CLI shim
- config.json / config.yaml — 示例配置
- dame_asc/ — 包实现
  - data/loader.py — manifest 读取与合成样本生成（支持 dict/list4）
  - augment/dcdir_bank.py — Mel-EQ Bank 实现（Option A）
  - models/experts/ — 占位 experts（passt, cnn）
  - models/fusion/dcf.py — 占位 DCF 融合器实现
  - models/factory.py — 构建专家/融合模块的工厂函数
  - pipeline.py — 推理流水线示例（集成增强）
  - losses/ — CE, consistency, reg 占位实现
  - utils.py — IO/日志帮助函数
- src/train/ — 占位训练脚本（stage1/stage2）
- src/infer/ — 推理脚本（TTA + per-device 回退）
- src/tools/ — CLI 工具（build_device_table）
- data/manifests/ — 示例 manifest 与映射文件
- tests/ — 基本单元测试

配置要点（核心字段）
- dataset.manifest: 指向 manifest 文件（JSON）
- dataset.manifest_format: "dict" 或 "list4"（list4 格式含义: [path, city, scene, device]）
- input: 可设置特征类型（示例: logmel 参数或 waveform）
- augment.dcdir: DCDIR 配置（mode: mel_eq_bank, bank_size, max_db, smooth_kernel, p）
- model.experts: 专家列表，示例：

```yaml
model:
  num_classes: 10
  experts:
    - name: passt
      pretrained: true
    - name: cnn
      pretrained: false
  fusion:
    name: dcf
    embed_dim: 128
    hidden: 256
    use_temperature: true
```

- loss: 主损失与正则

```yaml
loss:
  ce: {label_smoothing: 0.05}
  consistency: {enable: true, weight: 0.5, type: kl}
  reg: {gate_entropy_weight: 1e-3, dcdir_l2_weight: 1e-4}
```

推理策略（known device -> device-specific ckpt，否则回退）
- 推理脚本 `src/infer/infer.py` 会检查 `--device_ckpt_dir` 下是否存在 `device_{id}.ckpt` 或 `device_{id}/best.ckpt`。
- 若存在则标记为使用 stage-2 ckpt（占位：目前并未实际加载不同权重，后续可通过 `models/factory.py` 实现 ckpt 加载）。
- 支持 TTA（test-time augmentation）配置：`infer.tta.enable`、`infer.tta.num_crops`、`infer.tta.time_shift`（占位）。

必做的自检（建议的实验顺序）
1. 运行 `build_device_table`：计算每个 expert 在每个 device 上的准确率（CSV）。如果不同 device 上专家的排序发生交换，DCF 更可能带来收益。示例命令：

```cmd
python src/tools/build_device_table.py --val_manifest data/manifests/val.json --out_csv runs/per_device_table.csv
```

2. DCDIR（增强）先用随机 mel-EQ 做对照实验：若随机增强能带来提升，再尝试条件化 bank（DCDIR）；否则先检查增强强度/数据切分。

消融实验建议（论文/报告友好）
- Fusion: best single expert / avg ensemble / static learned weights / DCF / DCF+temperature
- DIR: none / random DIR / DIR bank (unconditioned) / DCDIR (device-conditioned)
- 建议组合实验矩阵覆盖上面两类的组合

如何替换为真实模型（路线）
- 在 `dame_asc/models/experts/` 中替换或新增真实实现（PyTorch）：实现 `predict(sample)` 返回 logits 或实现 `fit/load/save` 且在 `models/factory.py` 中注册。建议使用 PyTorch 并在 `requirements-ml.txt` 中列出額外依赖。
- 在 `src/train/` 中将占位训练流程替换为真实的 DataLoader、Batching、优化器、checkpointing 与分布式策略。

开发者提示与常见问题
- manifest 的 `scene` 字段必须在 `0..C-1`（C 为模型类别数）；占位脚本会把超出范围的 label 通过 modulo 映射（仅用于占位演示），真实训练时请保证标签正确。
- 若使用 YAML 配置，确保安装 PyYAML；否则使用 `config.json`。
- 占位实现的目的是保证接口与数据流无误，评估模块（DCDIR、DCF）逻辑可独立验证后，再替换模型权重部分。

联系方式与后续工作
- 如果你希望我继续：我可以
  - A) 把 `build_device_table` 扩展为输出 per-sample logits，用以训练 DCF（快速验证方案）;
  - B) 实现 `tools/fit_dcf_only.py`（冻结 experts、训练 DCF）;
  - C) 替换占位 experts 为 PyTorch 小模型并接入训练流程（需确认依赖）。

感谢使用本占位骨架 — 回馈问题或选择下一步实现即可，我会继续把功能变为可重复、可验证的完整实现。
