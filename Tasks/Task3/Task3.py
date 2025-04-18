import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import os 

current_dir = os.getcwd()

data_dir = os.path.join(current_dir, 'Data')

data = """building_id, mean_temp, std_temp, pct_above_18, pct_below_15
10000, 14.01233878811275, 6.367431059312565, 30.941014791508444, 55.542295034537624
10009, 11.000135812436373, 5.811144379826625, 16.6712734948236, 74.9723590310584
10014, 14.744169941950119, 7.037733284673848, 38.26367541377415, 52.80837116508215
10019, 14.735524480624482, 7.030325006703675, 38.14915412864569, 52.92926826787113
10029, 10.616037322820358, 6.317331938274926, 18.25563221896085, 74.51301795448481
10031, 12.507072852890545, 6.278432089100354, 24.044722033998173, 66.39513301711693
10051, 13.289039951277402, 5.999085063388632, 25.97693550756574, 62.859923608050536
10053, 11.366493551285709, 6.26121798185875, 19.9510754583921, 72.41052538787024
10056, 14.220114507861702, 6.179461157398302, 31.763454814173965, 57.06174975667784
10064, 12.71696893739585, 6.964227784263683, 28.79137124461432, 62.75688345539249
10075, 15.156939199079357, 6.44052034037085, 39.12088154756647, 47.45605511880576
10079, 15.094353507626135, 7.313911268349323, 41.90001451870493, 49.44974108309539
10080, 15.777740694240359, 7.280585752157965, 46.365765006711015, 44.711034476002
10082, 16.465720758630678, 6.713345052234242, 48.64349722630506, 40.56137689061685
10083, 15.639247995421403, 7.120808056609733, 44.855518923515284, 45.886354482120744
10084, 15.100584697661853, 7.1505418077486445, 40.90648998644782, 50.266526125583496
10085, 15.868862158668058, 7.192791728448739, 46.18303917834116, 44.72566696293788
10086, 14.391525374209257, 7.21561607319371, 37.25664572257129, 53.01884968583857
10087, 15.073205905031166, 7.275519953981684, 41.532405798190645, 49.89713190601896
10089, 13.989763514400206, 7.276278123379982, 35.45861191757374, 56.3640146392669
"""

df = pd.read_csv(StringIO(data))
df.columns = df.columns.str.strip()


plt.figure(figsize=(8, 4))
plt.hist(df['mean_temp'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Mean Temperatures')
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('Number of Buildings')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['building_id'], df['mean_temp'], marker='o', label='Mean Temp')
plt.plot(df['building_id'], df['std_temp'], marker='s', label='Std Dev')
plt.xticks(rotation=45)
plt.xlabel('Building ID')
plt.ylabel('Temperature (°C)')
plt.title('Mean & Std Temperature per Building')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['building_id'], df['pct_above_18'], marker='^', label='% Area > 18°C')
plt.plot(df['building_id'], df['pct_below_15'], marker='v', label='% Area < 15°C')
plt.xticks(rotation=45)
plt.xlabel('Building ID')
plt.ylabel('Percentage of Room Area')
plt.title('Thermal Comfort Zones')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("Average Mean Temp:", df['mean_temp'].mean())
print("Average Std Temp:", df['std_temp'].mean())
print("Buildings with >50% area >18°C:", (df['pct_above_18'] > 50).sum())
print("Buildings with >50% area <15°C:", (df['pct_below_15'] > 50).sum())



def plot_simulation_heatmaps(building_ids, data_dir='../Data', result_dir='solutions', cmap='magma'):
    """
    Plots multiple Jacobi simulation results side-by-side using masks.
    
    Parameters:
    - building_ids: list of int or str, IDs of the buildings to plot
    - data_dir: path to directory containing *_interior.npy and *_domain.npy
    - result_dir: path to directory containing *_solution.npy
    """
    num = len(building_ids)
    fig, axs = plt.subplots(1, num, figsize=(4 * num, 5), constrained_layout=True)
    if num == 1:
        axs = [axs]

    for ax, bid in zip(axs, building_ids):
        # Load interior mask and simulation result
        interior_mask = np.load(join(data_dir, f"{bid}_interior.npy"))
        result = np.load(join(result_dir, f"{bid}_solution.npy"))[1:-1, 1:-1]  # remove padding

        # Mask the non-interior areas (walls and outside)
        masked_result = np.ma.masked_where(~interior_mask, result)

        im = ax.imshow(masked_result, cmap=cmap, origin='upper', vmin=5, vmax=25)
        ax.set_title(f"Building {bid}", fontsize=10)
        ax.axis('off')

    # Add a single colorbar
    cbar = fig.colorbar(im, ax=axs, shrink=0.7, location='right')
    cbar.set_label("Temperature (°C)")
    plt.subplots_adjust(right=0.9)
    plt.show()


plot_simulation_heatmaps([10000, 10009, 10014, 10051], data_dir='Data', result_dir='Tasks/solutions')





