import seaborn as sns
import matplotlib.pyplot as plt

# 假设这是你的数据
training_times = [0.059, 0.00084, 0.079, 0.38, 0.041, 0.011]  # 训练时间
accuracies = [0.97, 0.90, 0.70, 0.90, 0.88, 0.70]  # 准确度
names = ['Our PatchTST', 'KNN', 'Logistic Regression',
         'Random Forest', 'Decision Tree', 'SVM']
colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']  # 每个点的颜色

# 创建一个Seaborn的图形风格
sns.set_theme(style="whitegrid")

# 创建散点图，使每个点更大
scatter = sns.scatterplot(x=training_times, y=accuracies, color='none', s=200)

# 为每个点指定颜色并绘制
for i, color in enumerate(colors):
    plt.scatter(training_times[i], accuracies[i], color=color, s=200)

# 在每个点上标注两行文本
for i, name in enumerate(names):
    # 假设每行文本内容分别为模型名称和一些额外信息
    line1 = name
    line2 = accuracies[i]  # 这里替换为你需要的第二行文本内容
    # 根据字体大小和文本内容调整偏移量
    offset = 0.01
    plt.text(training_times[i], accuracies[i] + offset,
             line1, fontsize=12, ha='center', va='bottom')
    plt.text(training_times[i], accuracies[i] - offset,
             line2, fontsize=12, ha='center', va='top')

plt.xlabel('Training Time')
plt.ylabel('Accuracy')

# 手动创建图例
handles = [plt.Line2D([0], [0], marker='o', color=color, label=name, markersize=10)
           for color, name in zip(colors, names)]
plt.legend(handles=handles, title='Models', loc='lower right')

# 显示图表
plt.show()
