##############################################
# Stefan Grulović (20150280) - Project part A
# 10/6/2019
# Part A is to build a program which can read a file
# that contains Formula 1 racing results and then filter,
# search and or calculate statistics from the data.
##############################################

# Libraries used for the program
import csv
import numpy as np
import datetime
import matplotlib.pyplot as plt
plt.rcdefaults()



################################################################################################################################################################################
# PART A
################################################################################################################################################################################

# Choice 1 read file from input and displays in table form
def read_display_input():
    fname = "partA_input_data.txt"
    try:
        file = open(fname)
        table = list(csv.reader(file, delimiter=','))
        line_count = 0
        table_separator = '{:<15s} {:<15s} {:<20s} {:<10s} {:<5s} {:<10s}'
        for row in table:
            if line_count == 0:
                print('____________________________________________________________________________________')
                print('                                RACE RESULTS')
                print('____________________________________________________________________________________')
                print(table_separator.format(*row))
                print('------------------------------------------------------------------------------------')
                line_count += 1
            else:
                print(table_separator.format(*row))
                line_count += 1
        print(f'\nProcessed {line_count} lines.')
        print('____________________________________________________________________________________')
    except IOError:
        print("Could not read file:", fname)

# Choice 2 asks for limit of laps and displays all the results from the file with lap lower than the amount provided
def search_limit_input():
    laps = input("Enter the limit of laps? ")
    try:
        int(laps)
        if int(laps) < 0:
            print("Number less than 0!")
        else:
            fname = "partA_input_data.txt"
            file = open(fname)
            table = list(csv.reader(file, delimiter=','))

            table[1:] = sorted(table[1:], key=lambda row: row[4], reverse=False)

            line_count = 0
            table_separator = '{:<15s} {:<15s} {:<20s} {:<10s} {:<5s} {:<10s}'

            for row in table:
                if line_count == 0:
                    print('____________________________________________________________________________________')
                    print('                         RACE RESULTS ( LAPS < ' + laps + ' )')
                    print('____________________________________________________________________________________')
                    print(table_separator.format(*row))
                    print('------------------------------------------------------------------------------------')
                    line_count += 1
                else:
                    if (row[4] < laps):
                        print(table_separator.format(*row))
                    line_count += 1
            print('____________________________________________________________________________________')

    except ValueError:
        print("Not a integer")


# Choice 3 reads input file, calculates the average lap times, displays it in table form and creates and output file
def average_lap_time():
    fname = "partA_input_data.txt"
    outputfname = "partA_output_data.txt"
    file = open(fname)
    table = list(csv.reader(file, delimiter=','))

    table[0].insert(6, "AVG_LAP_TIME")

    for row in table[1:]:
        timestamp = datetime.datetime.strptime(row[5], '%H:%M:%S.%f')
        # print(timestamp)
        time = timestamp.time()

        seconds = (time.hour * 60 + time.minute) * 60 + time.second
        average_time = seconds / float(row[4])
        avg_timestamp = datetime.timedelta(seconds=average_time)
        row.append(str(avg_timestamp))

    with open(outputfname, 'w+', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(table)

    output_file.close()
    print("Output file with data created!")

    line_count = 0
    table_separator = '{:<15s} {:<15s} {:<20s} {:<10s} {:<5s} {:<20s} {:<10s}'

    for row in table:
        if line_count == 0:
            print(
                '______________________________________________________________________________________________________________')
            print('                                       RACE RESULTS + AVG_LAP_TIME')
            print(
                '______________________________________________________________________________________________________________')
            print(table_separator.format(*row))
            print(
                '--------------------------------------------------------------------------------------------------------------')
            line_count += 1
        else:
            print(table_separator.format(*row))
            line_count += 1
    print(
        '______________________________________________________________________________________________________________')


# Choice 4 reads output file asks for one of the columns by which the data will be sorted either ascending or descending and displays it in table form
def sort_output_file():
    fname = "partA_output_data.txt"

    try:
        file = open(fname)
    except FileNotFoundError:
        print("Please first run Choice 3 in order for this to work!")
        return 0

    table = list(csv.reader(file, delimiter=','))

    print("__________________________________")
    print("             SORT BY")
    print("----------------------------------")
    print("[1] GRAND_PRIX")
    print("[2] DATE")
    print("[3] WINNER")
    print("[4] CAR")
    print("[5] LAPS")
    print("[6] TIME")
    print("[7] AVG_LAP_TIME")
    print("__________________________________")

    column = -1

    while (int(column) > 7) or (int(column) < 1):
        column = input("Chose a column: ")
        try:
            if (int(column) > 7) or (int(column) < 1):
                print("Choice doesnt exist!")
        except ValueError:
            print("Not a integer")

    asc_dsc = -1

    while (asc_dsc != "a") and (asc_dsc != "d"):
        asc_dsc = input('Ascending ("a") or Descending ("d")')
        if (asc_dsc != "a") and (asc_dsc != "d"):
            print("Option not available!")

    if asc_dsc == "a":
        reverse = False
    else:
        reverse = True

    if int(column) - 1 == 1:
        # 25 - Mar - 18
        table[1:] = sorted(table[1:], key=lambda row: datetime.datetime.strptime(row[1], '%d-%b-%y'))
    else:
        table[1:] = sorted(table[1:], key=lambda row: row[int(column) - 1], reverse=reverse)

    line_count = 0
    table_separator = '{:<15s} {:<15s} {:<20s} {:<10s} {:<5s} {:<20s} {:<10s}'

    for row in table:
        if line_count == 0:
            print(
                '______________________________________________________________________________________________________________')
            print('                                           RACE RESULTS')
            print(
                '______________________________________________________________________________________________________________')
            print(table_separator.format(*row))
            print(
                '--------------------------------------------------------------------------------------------------------------')
            # print(f'{", ".join(row)}')
            line_count += 1
        else:
            print(table_separator.format(*row))
            line_count += 1
    # print(f'\nProcessed {line_count} lines.')
    print(
        '______________________________________________________________________________________________________________')

# Choice 5 reads the input file calculates the total average lap time for each of the different drivers
def total_average_lap_time():
    fname = "partA_input_data.txt"
    try:
        file = open(fname)
        table = list(csv.reader(file, delimiter=','))
        unique_names_laps = {}

        for row in table[1:]:
            if row[2] not in unique_names_laps:
                unique_names_laps.update({row[2]: [[row[4], row[5]]]})
            elif row[2] in unique_names_laps:
                unique_names_laps[row[2]].append([row[4], row[5]])

        total_racer_laptimes = []

        names = []
        averages = []

        for name in unique_names_laps:
            seconds = 0
            total_average_time = 0
            num_of_races = 0
            for performance in unique_names_laps[name]:
                timestamp = datetime.datetime.strptime(performance[1], '%H:%M:%S.%f')
                # print(timestamp)
                time = timestamp.time()
                laps = performance[0]

                seconds = (time.hour * 60 + time.minute) * 60 + time.second
                average_time = seconds / float(performance[0])
                total_average_time += average_time
                num_of_races = num_of_races + 1

            total_average_time = total_average_time / num_of_races / 60

            names.append(name)
            averages.append(total_average_time)

            # print(averages)

            avg_timestamp = datetime.timedelta(minutes=total_average_time)
            #total_racer_laptimes.update({name : avg_timestamp})
            total_racer_laptimes.append([name, str(avg_timestamp), str(num_of_races)])

        # print(total_racer_laptimes)
        table_separator = '{:<25s} {:<25s} {:<15s}'

        print('________________________________________________________________')
        print('                 RACERS TOTAL AVERAGE LAP TIME')
        print('________________________________________________________________')
        # print(table_separator.format("NAME","TOTAL_AVG_LAP_TIME") )
        print(table_separator.format("NAME", "TOTAL_AVG_LAP_TIME", "NUM_OF_RACES"))
        print('----------------------------------------------------------------')

        for racer in total_racer_laptimes:
            # print( table_separator.format(racer, str(total_racer_laptimes[racer])))
            print(table_separator.format(racer[0], racer[1], racer[2]))

        print('________________________________________________________________')

        y_pos = np.arange(len(names))
        plt.bar(y_pos, averages, align='center', alpha=0.5)
        plt.xticks(y_pos, names, fontsize=8, rotation=-60)
        plt.ylabel('Average Lap Time in Minutes')
        plt.title('Racers Total Average Lap Time')

        plt.show()

        # print(unique_names_laps)
        # print(total_racer_laptimes)

    except IOError:
        print("Could not read file:", fname)


################################################################################################################################################################################
# MENU
################################################################################################################################################################################

def quit():
    print("__________________________________")
    print("               QUIT")
    print("----------------------------------")
    print("Program will exit, bye!")
    print("__________________________________")
    raise SystemExit


def error_invalid():
    print("__________________________________")
    print("               ERROR")
    print("----------------------------------")
    print("Invalid Choice, Please try again!")
    print("__________________________________")


def menu():
    menu = {"1": ('Read and Display contents from "partA_input_data.txt" '
                  '\n\t\t*needs "partA_input_data.txt" file', read_display_input),
            "2": ('Search contents by limit of laps '
                  '\n\t\t*Displayed ascending', search_limit_input),
            "3": ('Calculate the average lap time per race '
                  '\n\t\t*Adds 7th column to new file "partA_output_data.txt."', average_lap_time),
            "4": ('Sort by field, asc or dsc'
                  '\n\t\t*Needs file from Choice 3', sort_output_file),
            "5": ('Calculate the total average lap time per driver across all races'
                  '\n\t\t*Displayed as pop-up windows( x-axis: driver names. y-axis: average lap time in minutes)',
                  total_average_lap_time),

            "6": ("QUIT", quit),
            }

    print("__________________________________")
    print("               MENU")
    print("----------------------------------")
    for key in sorted(menu.keys()):
        print("[" + key + "]:  " + menu[key][0])
    print("__________________________________")

    choice = input("Make a Choice: ")
    menu.get(choice, [None, error_invalid])[1]()


########################################################################################
# MAIN PROGRAM
########################################################################################
while (True):
    menu()
