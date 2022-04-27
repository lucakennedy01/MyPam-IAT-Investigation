import configparser

config = configparser.ConfigParser()
config.read('config.ini')

width = 40

def build_options_menu():
    menu = {
        1: (str('Target Zone Radius' + (" " * (width - len("Target Zone Radius"))) + config.get("labelling", "radius"))),
        2: (str('Parsed Dataframe Size' + (" " * (width - len("Parsed Dataframe Size"))) + config.get("labelling", "size"))),
        3: (str('Search Directory' + (" " * (width - len("Search Directory"))) + config.get("labelling", "directory"))),
        4: (str('Draw Targets' + (" " * (width - len("Draw Targets"))) + str(bool(config.get("view", "targets"))))),
        5: (str('Draw Ideal Paths' + (" " * (width - len("Draw Ideal Paths"))) + str(bool(config.get("view", "lines"))))),
        6: (str('Lower Anomaly Detection Threshold' + (" " * (width - len("Lower Anomaly Detection Threshold"))) + config.get("anomalies","lower"))),
        7: (str('Upper Anomaly Detection Threshold' + (" " * (width - len("Upper Anomaly Detection Threshold"))) + config.get("anomalies","upper"))),
        8: (str('Number of Training Epochs' + (" " * (width - len("Number of Training Epochs"))) + config.get("autoencoder","epochs"))),
        9: (str('Reconstruction Dimensional Length' + (" " * (width - len("Reconstruction Dimensional Length"))) + config.get("autoencoder","reconstruction_dim"))),
        10: 'Go Back'
    }
    return menu

def print_edit_menu(menu):
    for key in menu.keys():
        print(key, '--', menu[key])

def edit_radius():
    radius = input("Enter new radius: ")
    config.set('labelling','radius','{r}'.format(r=radius))
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    print("New radius saved as: ", radius, " to config.ini")

def edit_size():
    size = str(input("Enter new size: "))
    config.set("labelling", "size", '{s}'.format(s=size))
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    print("New sample size saved as: ", size, " to config.ini")

def edit_directory():
    dir = str(input("Enter new directory: "))
    config.set("labelling", "directory", '{d}'.format(d=dir))
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    print("New directory saved as: ", dir, " to config.ini")

def edit_draw_targets():
    while(True):
        x = str(input("Draw Targets [y/n]: "))
        if (str(x).lower() in ["y", "yes", "true", "t"]):
            x = "True"
            config.set("view","targets", "{x}".format(x=x))
            with open('config.ini', 'w') as configfile:
                config.write(configfile)
            return
        elif (str(x).lower() in ["n", "no", "false", "f"]):
            x = ""
            config.set("view","targets", "{x}".format(x=x))
            with open('config.ini', 'w') as configfile:
                config.write(configfile)
            return
        else:
            print('Wrong input. Please yes or no\n')

def edit_draw_lines():
    while(True):
        x = str(input("Draw Ideal Paths [y/n]: "))
        if (str(x).lower() in ["y", "yes", "true", "t"]):
            x = "True"
            config.set("view","lines", "{x}".format(x=x))
            with open('config.ini', 'w') as configfile:
                config.write(configfile)
            return
        elif (str(x).lower() in ["n", "no", "false", "f"]):
            x = ""
            config.set("view","lines", "{x}".format(x=x))
            with open('config.ini', 'w') as configfile:
                config.write(configfile)
            return
        else:
            print('Wrong input. Please yes or no\n')

def edit_anomaly_threshold_lower():
    while(True):
        try:
            t = int(input("Enter new threshold: "))
            config.set("anomalies","lower","{a}".format(a = t))
            with open('config.ini','w') as configfile:
                config.write(configfile)
            return
        except ValueError:
            print("Please input an integer")

def edit_anomaly_threshold_upper():
    while(True):
        try:
            t = int(input("Enter new threshold: "))
            config.set("anomalies","upper","{a}".format(a = t))
            with open('config.ini','w') as configfile:
                config.write(configfile)
            return
        except ValueError:
            print("Please input an integer")

def edit_training_epochs():
    while(True):
        try:
            t = int(input("Enter new number of epochs: "))
            config.set("autoencoder","epochs","{e}".format(e = t))
            with open('config.ini','w') as configfile:
                config.write(configfile)
            return
        except ValueError:
            print("Please input an integer")

def edit_reconstruction_dim():
    while(True):
        try:
            t = int(input("Enter new dimensional width for decoder reconstruction: "))
            config.set("autoencoder","reconstruction_dim","{d}".format(d = t))
            with open('config.ini','w') as configfile:
                config.write(configfile)
            return
        except ValueError:
            print("Please input an integer")

def edit_menu():
    while(True):
        menu = build_options_menu()
        print_edit_menu(menu)
        option = ''
        try:
            option = int(input('Enter your choice: '))
        except:
            print('Wrong input. Please enter a number ...')
        #Check what choice was entered and act accordingly
        if option == 1:
            edit_radius()
        elif option == 2:
            edit_size()
        elif option == 3:
            edit_directory()
        elif option == 4:
            edit_draw_targets()
        elif option == 5:
            edit_draw_lines()
        elif option == 6:
            edit_anomaly_threshold_lower()
        elif option == 7:
            edit_anomaly_threshold_upper()
        elif option == 8:
            edit_training_epochs()
        elif option == 9:
            edit_reconstruction_dim()
        elif option == 10:
            print('Returning to main menu...')
            return()
        else:
            print('Invalid option. Please enter a number between 1 and ', len(menu))
    return

def main():
    edit_menu()

if __name__ == "__main__":
    main()
