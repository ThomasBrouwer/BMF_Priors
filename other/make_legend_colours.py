'''
Plot the legend for the four colours.
'''

import matplotlib.pyplot as plt


fig = plt.figure(figsize=(3,2))

plt.plot(range(5), range(5), label='Real-valued', color='r')
plt.plot(range(5), range(5), label='Nonnegative', color='b')
plt.plot(range(5), range(5), label='Semi-nonnegative', color='g')
plt.plot(range(5), range(5), label='Poisson', color='y')
plt.plot(range(5), range(5), label='Baseline', color='grey')


''' Set up the legend outside. '''
font_size_legend, number_of_columns, legend_box_line_width = 12, 7, 1
legend_line_width = 3
ax = fig.add_subplot(111)
legend_fig = plt.figure(figsize=(8.8,0.4))
legend = legend_fig.legend(*ax.get_legend_handles_labels(), 
                           loc='center', ncol=number_of_columns,
                           prop={'size':font_size_legend},
                           fancybox=True, shadow=False)
legend.get_frame().set_linewidth(legend_box_line_width)
plt.setp(legend.get_lines(), linewidth=legend_line_width)
    
plt.savefig('legend_colours.png', dpi=600)