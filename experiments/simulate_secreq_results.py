import matplotlib.pyplot as plt
import numpy as np
import os

# Simulated Macro F1 results based on typical TeasCo-FT vs Baseline performance on SecReq
scenarios = ['Train: CPN+ePurse\nTest: GPS', 'Train: CPN+GPS\nTest: ePurse', 'Train: ePurse+GPS\nTest: CPN']
lr_scores = [0.452, 0.481, 0.435]
rf_scores = [0.478, 0.512, 0.460]
teasco_scores = [0.541, 0.575, 0.528]

x = np.arange(len(scenarios))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, lr_scores, width, label='Logistic Regression', color='#e74c3c')
rects2 = ax.bar(x, rf_scores, width, label='Random Forest', color='#f39c12')
rects3 = ax.bar(x + width, teasco_scores, width, label='TeasCo-FT Cascade', color='#2ecc71')

ax.set_ylabel('Macro F1 Score', fontsize=12)
ax.set_title('Simulated Cross-Project Evaluation on SecReq Dataset\n(Macro F1 Scores)', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=10)
ax.set_ylim(0, 0.65)
ax.legend(loc='upper right', fontsize=10)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

# Save the plot
output_path = os.path.join(os.path.dirname(__file__), 'secreq_simulated_results.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graph successfully saved to: {output_path}")

# Print text summary
print("\n" + "="*60)
print(" SIMULATED MACRO F1 RESULTS - SecReq Cross-Project")
print("="*60)
print(f"{'Scenario':<25} | {'LR Base':<8} | {'RF Base':<8} | {'TeasCo-FT':<8}")
print("-" * 60)
for i in range(3):
    name = scenarios[i].replace('\n', ' -> ')
    print(f"{name:<25} | {lr_scores[i]:<8.3f} | {rf_scores[i]:<8.3f} | {teasco_scores[i]:<8.3f}")
print("-" * 60)
avg_lr = np.mean(lr_scores)
avg_rf = np.mean(rf_scores)
avg_tc = np.mean(teasco_scores)
print(f"{'AVERAGE':<25} | {avg_lr:<8.3f} | {avg_rf:<8.3f} | {avg_tc:<8.3f}")
print("="*60)
