Denne mappen inneholder data lastet ned i XML- og GRIB-format fra
yr.no, jfr retningslinjene på http://om.yr.no/verdata/xml/ og
http://om.yr.no/verdata/grib/. 

==================================================

Forarbeid:
----------

Filen noreg.txt er lastet ned fra yr.no:
$ wget http://fil.nrk.no/yr/viktigestader/noreg.txt

Filen noreg_viktige.txt er en filtrert versjon av noreg.txt hvor
alle steder med prioritet 99 er fjernet, jfr anbefalingen på
yr.no:
$ /store/gnu/bin/awk --field-separator='\t' '{ if ($4 != 99) print $0}' noreg.txt >noreg_viktige.txt

Likeledes inneholder noreg_viktige_namn.txt kun stedsnavn og nummer:
$ /store/gnu/bin/awk --field-separator='\t' '{print $1, $2}' noreg_viktige.txt >noreg_viktige_namn.txt

Værvarslene samles i undermapper under mappen "steder". Hver
mappe som skal inneholde værvarsel (i utgangspunktet alle løvnodene
i mappetreet) må inneholde en fil kalt "address.txt". Denne filen
skal inneholde en (og kun en) linje fra noreg.txt, som
spesifiserer sted og URL for varselet som skal lastes ned.

For de "enkle" stedene (ikke æøå i navnet og ingen undermapper)
ble address.txt automatisk generert (sed-biten fjerner DOS linjeskift):
$ for d in Bergen Drammen Oslo Stavanger Steinkjer Trondheim; do /store/gnu/bin/awk --field-separator='\t' "{ if (\$2 == \"$d\") print \$0}" noreg_viktige.txt | sed -e's/forecast\.xml.*$/forecast.xml/' >steder/$d/address.txt; done

For de andre stedene ble address.txt laget manuelt vha
klipp-og-lim-teknologi.

==================================================

Nedlasting:
-----------

Selve nedlastingen skjer vha skriptet "get-forecast.sh". 

Dette søker opp alle filene kalt address.txt og henter ut
XML-adressen derfra. Adressen endres til å hente timesvarsel
heller enn 6-timers. Varsel og wget-log lagres i de respektive
mappene.

Deretter lastes GRIB-filen for Nord-Europa ned og lagres i mappa
"GRIB".
