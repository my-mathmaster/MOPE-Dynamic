# MOPE-Dynamic


# 动态环境下的多目标路径规划仿真系统部署

## 系统要求

### 硬件要求

- 处理器: Intel Core i5 或同等性能及以上
- 内存: 8GB 或以上
- 显卡: 支持OpenGL 3.0及以上
- 存储空间: 至少100MB可用空间

### 软件要求

- 操作系统: Windows 10/11, macOS 10.15+, 或 Linux (Ubuntu 20.04+)
- Python 3.8 或更高版本
- pip (Python包管理工具)

## 安装步骤

### 1. 安装Python

如果您的系统尚未安装Python，请按照以下步骤安装:

#### Windows/macOS

1. 访问 [Python官网](https://www.python.org/downloads/)
2. 下载最新版本的Python安装包
3. 运行安装程序，确保勾选"Add Python to PATH"选项
4. 完成安装

#### Linux (Ubuntu)

```bash
sudo apt update
sudo apt install python3 python3-pip
```

### 2. 获取项目代码

您可以通过以下两种方式之一获取项目代码:

#### 方式一: 下载ZIP压缩包

1. 访问项目代码仓库
2. 点击"Download ZIP"按钮下载压缩包
3. 解压到您选择的目录

#### 方式二: 使用Git克隆

```bash
git clone https://github.com/your-repository/path-planning-simulation.git
cd path-planning-simulation
```

### 3. 安装依赖库

手动安装:

```bash
pip install pygame numpy matplotlib
```

### 4. 验证安装

运行以下命令验证安装是否成功:

```bash
python -c "import pygame; import numpy; print('依赖库安装成功')"
```

如果没有报错，表示安装成功。

## 运行仿真系统

### 启动仿真系统

在项目目录下运行以下命令:

```bash
python simulation.py
```

### 基本操作

1. **场景控制**:
   - 空格键: 暂停/继续仿真
   - N键: 切换到下一个场景
2. **可视化控制**:
   - S键: 显示/隐藏搜索节点
   - T键: 显示/隐藏DWA轨迹
3. **环境编辑**:
   - 鼠标左键: 添加静态障碍物
   - 鼠标右键: 移除静态障碍物

### 配置文件说明

系统使用JSON格式的配置文件定义场景，位于项目根目录下:

- `scene1.json`: 简单静态环境
- `scene2.json`: 动态障碍物环境
- `scene3.json`: 多目标点环境

您可以编辑这些文件或创建新的配置文件来自定义场景。配置文件结构如下:

```json
{
    "start": [1, 1],
    "end": [38, 28],
    "targets": [[10, 10], [30, 5], [20, 25]],
    "static_obstacles": [[5, 15], [6, 15], [7, 15]],
    "dynamic_obstacles": [
        {
            "position": [10, 20],
            "radius": 1,
            "movement": "linear",
            "speed": 0.5,
            "angle": 0.785
        }
    ]
}
```

## 常见问题解答

### Q1: 运行时报错"ModuleNotFoundError: No module named 'pygame'"

A: 这表明Pygame库未正确安装。请运行:

```bash
pip install pygame
```

### Q2: 如何添加自定义场景?

A: 在项目目录下创建一个新的JSON配置文件，然后修改代码中`config_files`列表包含您的新文件。