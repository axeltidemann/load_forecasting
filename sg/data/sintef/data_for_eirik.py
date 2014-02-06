import sg.data.sintef.userloads as ul
import sys

tf = ul.tempfeeder_nodup()
user_ids = tf.user_ids
for user in user_ids:
    loads = tf[user][:,0]
    idx = 0
    while loads.dates[idx].hour != 0:
        idx += 1
    while idx < len(loads) - 48:
        sys.stdout.write("%d %s False " % (user, loads.dates[idx].strftime("%Y-%M-%d")))
        for i in range(24):
            sys.stdout.write("%f " % loads[idx])
            idx += 1
        print ""
