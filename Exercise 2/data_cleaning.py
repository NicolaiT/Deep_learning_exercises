import csv

header = []
data = []

# Removing all rows that contains "?" and converts the data to their proper format for the linear regression
with open('data/auto.csv', 'r') as file:
	print('Cleaning auto.csv')
	reader = csv.reader(file)
	header = next(reader)

	for row in reader:
		invalid_row = False
		for row_data in row:
			if row_data == '?':
				invalid_row = True
				break

		row = [float(x) if x.isnumeric() else x for x in row]
		if invalid_row == False:
			data.append(row)

with open('data/clean_auto.csv', 'w') as file:
	print('Writing auto.csv')
	writer = csv.writer(file)
	writer.writerow(header)
	writer.writerows(data)
	print('Done!')

