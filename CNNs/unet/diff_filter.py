import csv
import os
import matplotlib.pyplot as plt
import numpy as np

datapath = '..\\..\\Data\\ModelDiff\\FilterDiff\\'

Filters = ['64', '32', '16', '8']
muscleNames = ['LM', 'LP', 'LQ', 'RM', 'RP', 'RQ']
Folders = [datapath + Filters[0], datapath + Filters[1], datapath + Filters[2], datapath + Filters[3]]

DiceValues = [[] for n in range(len(Filters))]
BestVlossRow = []

for f in Folders:
    Lossfile = os.listdir(f)[6]
    Lossfile = os.path.join(f, Lossfile)
    with open(Lossfile) as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        min_value = 1
        for row in csv_reader:
            if min_value > float(row[1]):
                min_value = float(row[1])
                rowNum = int(row[0])
    BestVlossRow.append(rowNum)

for i, f in enumerate(Folders):        # Weight Filenames
    muscles = os.listdir(f)[:6]        # Lists Muscles per Augment
    for m in muscles[:(len(muscleNames))]:
        filepath = os.path.join(f, m)
        with open(filepath) as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if int(row[0]) == BestVlossRow[i]:
                    Value = float(row[1])
                    DiceValues[i].append(Value)

DiceValues = np.transpose(DiceValues)
bar_width = 0.1

y_values1 = DiceValues[0]
y_values2 = DiceValues[1]
y_values3 = DiceValues[2]
y_values4 = DiceValues[3]
y_values5 = DiceValues[4]
y_values6 = DiceValues[5]

pos1 = np.arange(len(Filters))
pos2 = [x + bar_width for x in pos1]
pos3 = [x + bar_width for x in pos2]
pos4 = [x + bar_width for x in pos3]
pos5 = [x + bar_width for x in pos4]
pos6 = [x + bar_width for x in pos5]


fig, ax = plt.subplots()
ax.bar(pos1, y_values1, width=bar_width, label=muscleNames[0], color='darkgray')
ax.bar(pos2, y_values2, width=bar_width, label=muscleNames[1], color='lightcoral')
ax.bar(pos3, y_values3, width=bar_width, label=muscleNames[2], color='sandybrown')
ax.bar(pos4, y_values4, width=bar_width, label=muscleNames[3], color='khaki')
ax.bar(pos5, y_values5, width=bar_width, label=muscleNames[4], color='yellowgreen')
ax.bar(pos6, y_values6, width=bar_width, label=muscleNames[5], color='steelblue')

ax.set_xticks(pos3)
ax.set_xticklabels(Filters)


ax.set_title('Validation Dice Scores')
ax.set_xlabel('Filters')
ax.set_ylabel('Score')
ax.set_ylim(0.85, 1)  # Set the lower and upper bounds of the y-axis
ax.legend()
plt.savefig(datapath + 'FilterDiff.png')
ax.set_ylim(0, 1)
plt.savefig(datapath + 'FilterDiffBigScale.png')
