import serial
import sqlite3

#pitch values, side to side, temp

def processing(dd,arduData):
    #-180 - 0 - 180, -180 - 0 - 180
    #dd = [0,10]
    #6.6438

    pitch, roll, temp = [float(arduData[i]) for i in range(0,3)]

    temp = (temp - 7) * 1.49
    pitch = abs(pitch)/7.03
    roll = abs(roll)/15
    dd = dd /10

    #rating 60,20,15,5

    if (temp < 53):
        rating = 5 - (70 * dd + (1.5**pitch)/2.3 + (1.5**roll)/2.3 ) / 20
    else:
        rating = 5 - (30 * dd + 1.5 * (temp - 53) + (1.5**pitch)/2 + (1.5**roll)/2 ) / 20

    if rating < 0:
        rating = 0
    elif rating > 5:
        rating = 5

    return rating


def SqlReader(conn):
    cursor = conn.execute('SELECT max(sno) FROM data')
    max_id = cursor.fetchone()[0]
    cursor = conn.execute("SELECT * FROM data WHERE sno = " + str(max_id))
    for row in cursor:
        Sno, AccY, AccX, Temp = row
    return [AccY,AccX,Temp]
    #return ['30','50','26']

def serialWriter(rating):
    ser = serial.Serial('/dev/ttyACM0', 9600)
    ser.write(str(round(rating)))

def main(driver_distraction):

    conn = sqlite3.connect('fix.db')
    #conn = "lol"
    sqlData = SqlReader(conn)
    result = processing(driver_distraction,sqlData)
    #serialWriter(result)
    return round(result,2)

    conn.close()

if __name__ == "__main__":
    print(main(6))
