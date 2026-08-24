# MoonBit SafeTensors Toolkit

[English](README.md)

SafeTensors Toolkit 是一个原生 MoonBit CLI 与库，用于校验 SafeTensors 文件头、比较容器结构，并在不解释 tensor 数值的前提下执行确定性的整 tensor 分片、合并与重命名。

它实现 SafeTensors v1 容器结构：8 字节小端 header 长度、严格 UTF-8 JSON header、`__metadata__`、tensor 的 dtype/shape/range，以及连续 data buffer。当前 dtype 矩阵包含官方 F4/F6/F8 packed dtype，以及 BOOL、整数、F16/BF16/F32/F64、C64 和无符号整数类型。

## 快速开始

```bash
moon run src/cmd/sftk -- validate model.safetensors
moon run src/cmd/sftk -- info model.safetensors --format json
moon run src/cmd/sftk -- query model.safetensors 'prefix=encoder.,dtype=F16|F32,rank=2'
moon run src/cmd/sftk -- diff before.safetensors after.safetensors

# 默认仅展示计划，不写入文件。
moon run src/cmd/sftk -- split model.safetensors \
  --max-bytes 2GiB --output-dir shards

# 确认计划后，再显式写入。
moon run src/cmd/sftk -- split model.safetensors \
  --max-bytes 2GiB --output-dir shards --write
moon run src/cmd/sftk -- merge shards/model.safetensors.index.json \
  --output merged.safetensors --write
moon run src/cmd/sftk -- rename-prefix model.safetensors \
  --from-prefix encoder. --to-prefix model.encoder. \
  --output renamed.safetensors --write
```

`split` 输出 Hugging Face 风格的 JSON index，包含 `metadata.total_size` 与 `weight_map`；`merge` 读取该受限且严格校验的子集。已有输出文件时，必须同时传入 `--write` 与 `--force`。

最小 native 使用示例（只验证 header）：

```bash
moon run src/examples/validate --target native -- model.safetensors
```

## 命令与退出码

| 命令 | 行为 | 退出码 |
|---|---|---|
| `validate PATH` | 有界读取 header 并完成结构校验 | 有效为 `0`，无效为 `1` |
| `info PATH` | 仅读取 header，输出 tensor 数量与 dtype 分布 | 成功为 `0` |
| `query PATH SELECTOR` | 按选择器稳定查询 tensor 声明，不读取 payload | 成功为 `0` |
| `diff LEFT RIGHT` | 比较两个容器的 tensor/metadata 结构，不读取 payload | 相等为 `0`，不同为 `1` |
| `split …` | 生成整 tensor 的贪心分片计划；`--write` 执行有界 range copy | 成功为 `0` |
| `merge …` | 校验 index/shard 覆盖后输出规范化合并文件 | 成功为 `0` |
| `rename-prefix …` | 只改 tensor 名称，payload range 保持不透明且逐字节复制 | 成功为 `0` |

参数错误返回 `2`；文件 I/O 与发布冲突返回 `3`。加上 `--format json` 可取得机器可读结果，诊断信息写入 stderr。

## 安全与兼容性边界

- header 长度按无符号小端解析，默认上限为 100 MiB。
- JSON 解析器拒绝重复键、非法字符串与 Unicode 转义、非法数字、过深嵌套及不支持的 tensor 字段。
- 在复制数据前检查 shape 乘法、dtype 位宽、range、空洞、重叠与最终 data-buffer 覆盖。
- native 检查仅读取 8 字节前缀和有界 header；split/merge 使用固定大小 buffer（默认 1 MiB）复制数据，不解码 tensor 数值。
- split、rename 与 merge 会在发布前重算源 payload 指纹，能发现同大小 payload 替换；该指纹不是密码学认证，也不是操作系统级 compare-and-swap。
- split 先写临时文件，先发布 shard、最后发布 index；因此已发布 index 不会引用未发布 shard。但跨文件的全局原子性和进程中断恢复尚未保证。
- `merge` 按规范 tensor-name 顺序输出；每个 tensor 名称对应的字节会被保留，但生成文件不保证与分片前源文件逐字节相同。

严格 index reader 只接受 `metadata.total_size` 与 `weight_map`，并要求安全 shard basename；第三方 index 扩展会被拒绝。

## 验证

```bash
moon fmt --check
moon check --target all --deny-warn
moon build --target all --deny-warn
moon test --target all --deny-warn
bash scripts/cli_e2e.sh
PYTHON_BIN=python3 bash scripts/python_interop.sh
moon info
```

本地测试覆盖严格 prefix/JSON/layout、结构 Diff、确定性 plan/index/merge，以及检查每个命名 tensor 拷贝字节的 native split→merge 集成测试。`scripts/cli_e2e.sh` 在运行时生成原创小型 fixture，覆盖全部七个 CLI 命令，包括 `query` 与 `rename-prefix`。

外部互操作固定使用 `safetensors==0.6.2` 与 NumPy `2.2.6`。当前已覆盖小型 U8 fixture；完整 dtype 与大文件矩阵仍是后续发布工作。

## 设计材料

- [整体设计](docs/整体设计.md)
- [项目申请书](docs/项目申请书.md)

## 许可证

Apache-2.0。
