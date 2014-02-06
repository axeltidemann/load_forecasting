The data in this folder are those used in the EUNITE 2001 load forecasting competition. These have subsequently also been applied by other forecasting studies (e.g. T. Rashid and T. Kechadi, A Practical Approach for Electricity Load Forecasting, World Academy of Science, Engineering and Technology 5 2005).In the competition, the 1997 and 1998 data were used as training sets, while the competition used data from January 1999. 

See web page for more info:
http://neuron.tuke.sk/competition/index.php

* Web page of competition winners:
http://www.csie.ntu.edu.tw/~cjlin/papers.html
* Chang, Chen & Lin. EUNITE Network Competition: Electricity Load Forecasting:
http://www.csie.ntu.edu.tw/~cjlin/papers/euniteelf.ps.gz
Also saved in this directory as winner_model_article.pdf.

The data were preprocessed as follows:

* In Excel, all dates were formatted as ISO (YYYY-MM-DD).  

* All line endings were changed using mac2unix.

* The data in Holidays.xls was manually transformed so all dates were in a row.
 
* Temperature data for 1997 and 1998 (from "competition" folder) were manually opened in Excel, the two years were concatenated, the date format was set to ISO YYYY-MM-DD, and the file was saved as temperatures.csv.

* temperature.csv and temperature_1999.csv were modified replacing ',' with '.' as decimal separator. 

* Load data for 1997 and 1998 (from "competition" folder) were manually concatenated and saves as loads.csv.

* Dates for temperature 1999 were reformatted using the function _reformat_date_jan_1999 in import_csv_to_sqlite.py.

