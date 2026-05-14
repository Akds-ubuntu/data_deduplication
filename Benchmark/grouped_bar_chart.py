import matplotlib.pyplot as plt
import numpy as np

labels = ["SimHash", "MinHash", "Semantic"]
quora_f1 = [0.4455, 0.7042, 0.7167]
code_f1 = [0.1618, 0.4512, 0.0420]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width / 2, quora_f1, width, label="Quora (Text)", color="#3498db")
rects2 = ax.bar(
    x + width / 2, code_f1, width, label="CodeXGLUE (Code)", color="#e74c3c"
)

ax.set_ylabel("F1-Score")
ax.set_title("Сравнение эффективности алгоритмов дедупликации")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.show()