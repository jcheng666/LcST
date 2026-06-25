# LCSTM-3stage

LCSTM-3stage 是基于时空窗口建模的交通预测与缺失值补全实验代码库。主代码位于 `src/`，实验脚本位于 `scripts/`，仓库根目录只保留项目配置和说明文档。

## 默认辅助节点策略

当前主版本默认使用邻居辅助节点池：

- `--n_aux 16`
- `--aux_neighbor_order topological`
- `--aux_neighbor_fill higher_order`

含义如下：

- `topological`：候选邻居先按有向邻接图的拓扑秩排序，同秩时按节点编号排序。
- `higher_order`：若 1-hop 邻居不足 `n_aux`，依次补入 2-hop、3-hop 邻居；仍不足时再重复已有候选。

如需复现实验对照，可以显式切回旧策略：

```bash
--aux_neighbor_order index --aux_neighbor_fill repeat_1hop
```

## 运行提示

`pyproject.toml` 已声明本 README 为项目说明文件。做轻量语法检查时可以使用：

```bash
uv run --no-project python -m compileall -q src scripts
```

正式实验参数以 `experiments.log` 首行 `Namespace(...)` 或日志目录中的 `snapshot/args.json` 为准。
