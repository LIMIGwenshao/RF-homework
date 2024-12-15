import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定义 PINN 模型
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 50),  # 增加层大小
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 3)  # 输出 Ey, Hx, Hz
        )

    def forward(self, x):
        return self.hidden(x)

# 定义损失函数
def loss_function(model, x, boundary_conditions, weights, omega, mu_0, epsilon_0):
    preds = model(x)
    Ey, Hx, Hz = preds[:, 0], preds[:, 1], preds[:, 2]

    # 计算 ∇×E 和 ∇×H，并使用 retain_graph=True
    curl_E = torch.autograd.grad(Ey, x, grad_outputs=torch.ones_like(Ey), create_graph=True)[0][:, 1]  # dEy/dz
    curl_H = torch.autograd.grad(Hx, x, grad_outputs=torch.ones_like(Hx), create_graph=True)[0][:, 0]  # dHx/dx

    # 定义 L_p1
    L_p1 = (1 / (2 * len(x))) * torch.norm(curl_E + 1j * omega * mu_0 * Hz, p='fro')**2

    # 定义 L_p2
    L_p2 = (1 / len(x)) * torch.norm(curl_H - 1j * omega * epsilon_0 * Ey, p='fro')**2

    # 边界条件损失
    L_b = (1 / len(boundary_conditions)) * sum(
        torch.norm((model(coords) - values), p='fro')**2 for coords, values in boundary_conditions
    )

    total_loss = weights[0] * L_p1 + weights[1] * L_p2 + weights[2] * L_b

    return total_loss

# 初始化模型和优化器
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 降低学习率

# 生成随机采样点
num_samples = 2000  # 增加样本数量
x = torch.rand(num_samples, 2, requires_grad=True) * torch.tensor([22.86, 50.0])  # 规定 x 和 z 的范围

# 设置边界条件
linspace_100 = np.linspace(0, 22.86, 100)
boundary_conditions = [
    (torch.tensor([[x_val, 0.0] for x_val in linspace_100], dtype=torch.float32),
     torch.tensor([[0.0, 0.0, 0.0]] * 100, dtype=torch.float32)),
    (torch.tensor([[x_val, 50.0] for x_val in linspace_100], dtype=torch.float32),
     torch.tensor([[0.0, 0.0, 0.0]] * 100, dtype=torch.float32)),
    (torch.tensor([[0.0, z_val] for z_val in np.linspace(0, 50.0, 100)], dtype=torch.float32),
     torch.tensor([[10.0 * np.sin((3.1415 / 22.86) * z_val), 0.0, 0.0] for z_val in np.linspace(0, 50.0, 100)], dtype=torch.float32)),
    (torch.tensor([[22.86, z_val] for z_val in np.linspace(0, 50.0, 100)], dtype=torch.float32),
     torch.tensor([[10.0 * np.sin((3.1415 / 22.86) * z_val), 0.0, 0.0] for z_val in np.linspace(0, 50.0, 100)], dtype=torch.float32)),
]

weights = [1.0, 1.0, 1.0]  # 权重设置
omega = 1.0
mu_0 = 1.0
epsilon_0 = 1.0

# 用于记录损失值
loss_values = []

# 训练模型
num_epochs = 50000  # 增加Epoch数量
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_function(model, x, boundary_conditions, weights, omega, mu_0, epsilon_0)
    loss.backward(retain_graph=True)
    loss_values.append(np.log(loss.item()))  # 对损失值取对数
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 绘制损失值折线图
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Log(Loss)', color='b')
plt.xlabel('Epoch')
plt.ylabel('Log Loss Value')
plt.title('Log Loss Value Over Epochs')
plt.legend()
plt.grid()
plt.show()

# 在整个二维网格上进行预测
x_values = np.linspace(0, 22.86, 100)
z_values = np.linspace(0, 50.0, 100)
X, Z = np.meshgrid(x_values, z_values)
grid_points = np.vstack([X.ravel(), Z.ravel()]).T

grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
with torch.no_grad():
    predictions = model(grid_points_tensor)

Ey, Hx, Hz = predictions[:, 0].reshape(X.shape), predictions[:, 1].reshape(X.shape), predictions[:, 2].reshape(X.shape)

# **理论结果的计算**
theoretical_Ey = 10.0 * np.sin((3.1415 / 22.86) * Z)  # 理论结果示例
theoretical_Hx = Ey.detach().numpy() * 0.5  # 这里可以设定合理的理论逻辑
theoretical_Hz = Hz.detach().numpy() * 0.5  # 这里可以设定合理的理论逻辑

# 计算误差
error_Ey = Ey.detach().numpy() - theoretical_Ey
error_Hx = Hx.detach().numpy() - theoretical_Hx
error_Hz = Hz.detach().numpy() - theoretical_Hz

# 计算误差度量
mse_Ey = np.mean(error_Ey**2)
mse_Hx = np.mean(error_Hx**2)
mse_Hz = np.mean(error_Hz**2)

print(f'MSE for Ey: {mse_Ey:.4e}, Hx: {mse_Hx:.4e}, Hz: {mse_Hz:.4e}')

# 创建一个 3 行 2 列的画布
fig, axs = plt.subplots(3, 2, figsize=(15, 18))

# 绘制 Ey 的结果
contour1 = axs[0, 0].contourf(X, Z, Ey.detach().numpy(), levels=50, cmap='viridis')
axs[0, 0].set_title('Electric Field (Ey) Distribution (PINN)')
axs[0, 0].set_xlabel('X Position')
axs[0, 0].set_ylabel('Z Position')
fig.colorbar(contour1, ax=axs[0, 0], label='Ey Value')

contour1_theory = axs[0, 1].contourf(X, Z, theoretical_Ey, levels=50, cmap='viridis')
axs[0, 1].set_title('Electric Field (Ey) Distribution (Theoretical)')
axs[0, 1].set_xlabel('X Position')
axs[0, 1].set_ylabel('Z Position')
fig.colorbar(contour1_theory, ax=axs[0, 1], label='Ey Value')

# 绘制 Hx 的结果
contour2 = axs[1, 0].contourf(X, Z, Hx.detach().numpy(), levels=50, cmap='viridis')
axs[1, 0].set_title('Magnetic Field (Hx) Distribution (PINN)')
axs[1, 0].set_xlabel('X Position')
axs[1, 0].set_ylabel('Z Position')
fig.colorbar(contour2, ax=axs[1, 0], label='Hx Value')

contour2_theory = axs[1, 1].contourf(X, Z, theoretical_Hx, levels=50, cmap='viridis')
axs[1, 1].set_title('Magnetic Field (Hx) Distribution (Theoretical)')
axs[1, 1].set_xlabel('X Position')
axs[1, 1].set_ylabel('Z Position')
fig.colorbar(contour2_theory, ax=axs[1, 1], label='Hx Value')

# 绘制 Hz 的结果
contour3 = axs[2, 0].contourf(X, Z, Hz.detach().numpy(), levels=50, cmap='viridis')
axs[2, 0].set_title('Magnetic Field (Hz) Distribution (PINN)')
axs[2, 0].set_xlabel('X Position')
axs[2, 0].set_ylabel('Z Position')
fig.colorbar(contour3, ax=axs[2, 0], label='Hz Value')

contour3_theory = axs[2, 1].contourf(X, Z, theoretical_Hz, levels=50, cmap='viridis')
axs[2, 1].set_title('Magnetic Field (Hz) Distribution (Theoretical)')
axs[2, 1].set_xlabel('X Position')
axs[2, 1].set_ylabel('Z Position')
fig.colorbar(contour3_theory, ax=axs[2, 1], label='Hz Value')

plt.tight_layout()
plt.show()
