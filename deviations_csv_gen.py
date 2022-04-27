from anomaly_detection import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

config = configparser.ConfigParser()
config.read('config.ini')

dir_path = str(config.get("labelling", "directory"))
radius = int(config.get("labelling", "radius")) #search radius for assigning points to targets
size = int(config.get("labelling", "size"))
targetFlag = bool(config.get("view", "targets"))
pathFlag = bool(config.get("view", "lines"))
lower_threshold = int(config.get("anomalies","lower"))
upper_threshold = int(config.get("anomalies", "upper"))

def create_avdxy_csv(avg_dxy):
    avg_dxy_list = list(avg_dxy)
    head = list([size, radius])
    content = head + avg_dxy_list
    #write to a new csv
    with open("deviations/deviations-{s}-{r}.csv".format(s=size, r = radius), 'w', encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(content)

def get_dxy():
    file_count = count_files(dir_path)
    targets = build_targets()
    avg_dxy = np.zeros(file_count)
    anomalies = list()

    for i in range(1, file_count+1):
        df = get_data(i)
        df['Target'] = label_target(df, targets)
        avg_dxy[i-1] = round(avg_distance(df, targets), 2)
        print("Sample {i} avdxy: {a}".format(i = i, a = avg_dxy[i-1]))

    return avg_dxy

def main():
    dxy = get_dxy()
    create_avdxy_csv(dxy)

if __name__ == "__main__":
    main()
