# 重构计划：将 FEM 从项目中分离

## 当前问题
- FEM 框架代码被直接拷贝到项目中（不是 fork）
- 无法同步上游 FEM 更新
- 项目职责不清晰

## 重构目标
创建独立的 `fpga-placement` 项目，FEM 作为外部依赖

## 新项目结构

```
fpga-placement/                    # 项目根目录（重命名）
├── external/                      # 外部依赖
│   └── FEM/  (git submodule)     # 原始 FEM 框架
├── fpga_placement/                # 我们的包
│   ├── __init__.py
│   ├── placer.py                 # FpgaPlacer (RapidWright接口)
│   ├── objectives.py             # 目标函数
│   ├── drawer.py                 # 可视化
│   ├── legalizer.py              # 合法化
│   ├── router.py                 # 布线
│   └── utils.py                  # 工具函数
├── tests/
│   └── test_fpga_placement.py
├── examples/                      # 示例代码
├── vivado/                        # FPGA 设计文件
├── docs/                          # 文档
├── .gitmodules                    # submodule 配置
├── pyproject.toml                 # 标准 Python 包配置
├── setup.py                       # 兼容性
├── requirements.txt               # 依赖列表
├── .gitignore
└── README.md
```

## 重构步骤

### 1. 备份当前的 FEM 修改
```bash
# 在新分支上操作
git checkout xw/refactor

# 确认 FEM 已经清理干净（已完成）
git diff FEM/
```

### 2. 移除 FEM 目录
```bash
# 删除 FEM 目录（保留在 git 历史中）
git rm -r FEM/
git rm -r benchmarks/  # FEM 的 benchmark
git rm -r examples/    # FEM 的 examples（如果有的话）
```

### 3. 添加 FEM 为 submodule
```bash
# 创建 external 目录
mkdir external

# 添加原始 FEM 仓库为 submodule
git submodule add https://github.com/Fanerst/FEM.git external/FEM

# 初始化并更新 submodule
git submodule update --init --recursive
```

### 4. 更新 Python 路径引用
修改所有文件中的导入：
- `from FEM import ...` → `from external.FEM import ...`
- 或配置 `sys.path.append('external')`

### 5. 创建标准 Python 包配置文件
- `pyproject.toml` - 现代 Python 包配置
- `setup.py` - 向后兼容
- `requirements.txt` - 依赖管理

### 6. 更新文档
- 更新 README.md 说明新架构
- 添加安装和使用说明

## 使用方式

### 克隆项目
```bash
git clone --recursive <your-repo-url>
# 或
git clone <your-repo-url>
cd fpga-placement
git submodule update --init --recursive
```

### 安装
```bash
# 开发模式安装
pip install -e .

# 或直接安装依赖
pip install -r requirements.txt
```

### 更新 FEM 到最新版本
```bash
cd external/FEM
git pull origin main
cd ../..
git add external/FEM
git commit -m "Update FEM to latest version"
```

## 优势
1. ✅ FEM 代码完全分离，不污染项目
2. ✅ 可以轻松更新 FEM 到最新版本
3. ✅ 项目职责清晰：专注于 FPGA placement
4. ✅ 标准的 Python 包结构，易于分发
5. ✅ 可以独立发布为 PyPI 包
