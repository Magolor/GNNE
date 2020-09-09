# README.md

## Demo

`python3 batchrun.py`: 运行完整实验流程并在`./workspace/`下生成可视化结果。注意运行完整实验流程开始时会清空以前实验的全部数据。

`python3 batchrun.py --report-only`: 若`./workspace/`下已存在实验结果，可在`./workspace/`下生成可视化结果而无需重新运行。

## 全局参数

batchrun.py: `RANGE`, 需要运行的所有参数`p`
batchrun.py: `GPU`, 使用的GPU数量
batchrun.py: `ITERATION`, 每个`p`重复实验次数

每个点有10个feature, GNN为3层10-32-64, GNNE为3层10-32-64. feature采用one-hot形式, 具体地, 前4位表征1, 其后4位表征0, 最后2位恒为1. 即`f=1`指`[1,1,1,1,0,0,0,0,1,1]`, `f=0`指`[0,0,0,0,1,1,1,1,1,1]`. 用`f=2`表示所有feature的点.

<br/>

## 运行逻辑

- batchrun.py: RunInstances(): 对每个`p`进行实验

  - train.py: syn_task4(): 树环图任务

    - gengraph.py: gen_syn4(): 生成树环图
        
  - train.py由于--graph-only而中止

  - 对于每个ITERATION:

    - 对于每个`p`并行运行(`GPU`个一组)

      - train.py: syn_task4(): 树环图任务(已存在图)

        - train.py: train_node_classifier(): 训练GNN
        
    - 对于每个`p`

      - 生成GNN统计数据(`f=0,f=1,f=2`)

    - 对于每个`p`并行运行(`GPU`个一组)

      - explainer_main.py

        - explainer/explain.py: Explainer.stats(): 对`f=0`和`f=1`每个点训练GNNE, 返回预测结果

    - 对于每个`p`

      - 生成GNNE统计数据(`f=0,f=1,f=2`)

- batchrun.py: Report()

  - 根据GNN和GNNE重复实验的统计数据生成报告, 在每个`ITERATION`结束时都更新了统计数据, 因此运行过程中也可以使用Report查看当前为止的结果。

<br/>

## 修改/重要代码

首先去除了原代码中的绝大部分调试输出，为GNNE预测过程添加了`tqdm`方便查看GNNE训练进度。

### explainer/explain.py

- Explainer.stats(): 通过复制Explainer.explain_nodes_gnn_stats()并修改得到。预测一个点集内的所有点并返回预测结果序列。

- Explainer.make_pred_real(): 这是原作者手写暴力判断每条边是否为结构上的边的代码。因此如果要更换结构类型，需要手动修改这里的判断条件。

### utils/featgen.py

- BinomialFeatureGen(): 生成一个二项分布的feature, 每个点有`p`的概率`f=1`。构造方式为增量式: 给定seed即确定一个节点数长的排列，排列中的前`int(p*num)`个节点`f=1`.

- CorrelatedFeatureGen(): 生成一个与structure关联的feature, 有`n`个点中`m`个点为结构点, 则有`m`个`f=1`的点，其中恰好有`int(p*m)`个为结构点。构造方式为增量式: 给定seed即确定一个结构点和非结构点的排列，两个排列中`int(p*m)`和`m-int(p*m)`个节点`f = 1`.

### configs.py

-  arg_parse(): GNN参数设置

### explainer_main.py

-  arg_parse(): GNNE参数设置

- 添加了`--stat`选项的处理，用于对`f=0`和`f=1`分别训练GNNE进行预测并生成预测结果。

### gen_graph.py

- gen_syn4(): 在这里进行或其他gen_syn中修改以生成其他图。

### train.py

- syn_task4(): 修改了生成图的流程, 在生成图以后在这里进行feature的生成。

- train_node_classifier(): 训练GNN, 修改了返回结果使得`f=0`和`f=1`的预测结果分别返回。

<br/>

其余部分几乎没有用到/几乎没有修改。

<br/>

## 文件操作

- 首先清空`./workspace/`, `./log/`, `./ckpt`这些工作目录

- 生成图: `./workspace/graph.pkl`

- 生成图相关数据:
  - `./workspace/struct_nodes.pkl`记录一个mask，其中所有结构点为`True`。
  - `./workspace/feat_nodes_p=0.00.pkl`记录一个mask，其中所有`f=1`的点为`True`。`0.00`可以替换为任何在`RANGE`中存在的两位小数。
  - `./workspace/labeled_nodes_p=0.00.pkl`记录一个mask，其中所有label为1的点为`True`。

- 训练GNN:
  - `./workspace/GNN-p=0.00-prediction.pkl`输出GNN的预测结果，一个长为节点数的序列，每个值为为0/1表示二分类结果。
  - `./workspace/GNN-p=0.00-f=0-performance.tmp`一个序列，记录GNN在`f=0`上节点预测结果指标: [True Positive, False Positive, True Negative, False Negative]。`f=0`可以替换为`f=1`表示`f=1`上节点预测结果指标，`f=2`表示所有节点预测结果指标。
  - `./workspace/GNN-p=0.00-f=0-performance.pkl`综合多组实验的结果指标: 每个指标是一个字典，`'ACC'`表示Accuray，`'PRE'`表示Precision，`REC`表示Recall。

- 预测目标:
  - `./workspace/nodes-p=0.00-f=0.tmp`一个节点编号序列，记录需要GNNE预测的节点中所有`f=0`的节点。注意不存在`f=2`。

- 训练GNNE:
  - `./workspace/GNNE-p=0.00-f=0.stat`一个字典，记录GNNE预测结果的统计数据，其中`'data'`关键字里包含了[每条边GNNE预测结果，每条边是否为结构边的真正答案]，`'AUROC'`关键字里用前述数据计算了AUROC值（然而最后并没有用到）。
  - `./workspace/GNNE-p=0.00-f=0-performance.pkl`综合多组实验的结果指标: 每个指标是一个字典，`'AUROC'`表示AUROC值，`'TOPACC'`表示采用最佳阈值时的Accuracy，`'TOPTHRES'`表示前述最佳阈值，`'0.7ACC'`;`'0.8ACC'`;`'0.9ACC'`分别表示对应阈值下的Accuracy。

- 运行实验结束后删除临时文件
  - `./workspace/graph.pkl`
  - `./workspace/struct_nodes.pkl`
  - `./workspace/feat_nodes_p=0.00.pkl`等
  - `./workspace/labeled_nodes_p=0.00.pkl`等
  - `./workspace/GNN-p=0.00-f=0-performance.tmp`等

- 生成报告
  - `./report/`内的图片: 对于每个`p`，每个`f=0/1/2`，生成GNN的`'ACC'`, `'PRE'`, `REC`和GNNE的`'AUROC'`, `'TOPACC'`, `'TOPTHRES'`, `'0.7ACC'`, `'0.8ACC'`, `'0.9ACC'`。但是，`p=1.00`时`f=0`数据不存在；`p=0.00`时`f=1`数据不存在；`f=1`时GNN的`PRE`必为`100%`而无意义等，没有作图。

## 版本

### v1.0.1
- 修复了若干语法错误。
- 增加了`batchrun.py`中`--gnn-only`开关。
- 轻微修改了GNN的参数。

### v1.0.2
- `explainer/explain.py`中的函数`stats`传入`make_pred_real`处发生了修改，需要针对structure进行适应性修改！
- 轻微修改了GNN的参数。

### v1.1.0
- 增加了`batchrun.py`中`--syn-type`开关: 调整生成图的结构类型。前缀`ba`表示BarabasiAlbert随机图($n=1023$,$d=5$)，`tree`表示完全二叉树($n=1023$)；后缀`cycle`表示六元环，`grid`表示$3 \times 3$网格，`house`表示房子图。
- 增加了`batchrun.py`中`--feat-gen`开关: 切换feature生成方式。`Binomial`表示概率随机(实验一)；`Correlated`表示关联性分布(实验二)；`CorrelatedXOR`表示关联性分布，但feature与label关系为XOR而非OR(实验三)；`Independent`表示概率随机，但feature不影响label(实验一)；`Const`表示feature为常数$1$。
- 增加了`LOAD`参数，提高程序并行效率。
- 修正了若干错误。

### v1.1.1
- 调整了若干运行逻辑

<br/>
