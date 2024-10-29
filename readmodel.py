import torch

# 指定模型文件路径
model_path = 'dqn_model.pth'

try:
    # 加载模型的状态字典
    model_data = torch.load(model_path, weights_only=True)

    # 检查加载的数据类型
    if isinstance(model_data, dict):
        print("模型文件包含以下内容:")
        for key, value in model_data.items():
            print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else type(value)}")
            # 打印具体的数据，选择打印前5个元素
            if isinstance(value, torch.Tensor):
                print(f"具体数据（前5个元素）: {value[:5]}")  # 打印前5个元素
    else:
        print("模型文件内容不是字典，内容类型:", type(model_data))

except FileNotFoundError:
    print(f"文件 {model_path} 不存在。")
except Exception as e:
    print(f"加载模型时发生错误: {e}")
