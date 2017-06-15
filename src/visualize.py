import matplotlib.pylab as plt

def plot_bmi_curves(data_dic):
	ix = 0
	for k in data_dic.keys():
		bdate = data_dic[k]['bdate']
		if data_dic[k]['gender']:
			col='b'
		else:
			col='r'
		if 'vitals' in data_dic[k]:
			if 'BMI' in data_dic[k]['vitals']:
				for (date1, bmi) in data_dic[k]['vitals']['BMI']:
					age = (date1 - bdate).days / 365.0
					if bmi>100: 
						continue
					plt.scatter(age, bmi, s=0.1, color=col, alpha=0.5)
					ix += 1
		if ix > 50000:
			break		
	plt.show()
