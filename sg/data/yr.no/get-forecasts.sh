#!/bin/bash

find=/store/gnu/bin/find
awk=/store/gnu/bin/awk
sed=/store/gnu/bin/sed
date=/store/gnu/bin/date
wget=/opt/pkg/bin/wget

BASE_DIR="$HOME/sg-shared/data/yr.no"
GRIB_DIR="$BASE_DIR/GRIB"
PLACES_DIR="$BASE_DIR/steder"
ADDRESSES="address.txt"
FORECAST_FILE="forecast_hour_by_hour.xml"
NOW=`$date --iso-8601=hours`
OUTPUT_FILE="${NOW}_forecast_hour_by_hour.xml"
LOG_FILE="wget_log.txt"

echo -n "Retrieving hourly forecasts for $NOW:"
$find "$PLACES_DIR" -type f -name "$ADDRESSES" | while read ADDRESS_FILE; do
    PLACE_DIR=`dirname "$ADDRESS_FILE"`
    PLACE=`echo $PLACE_DIR | $awk --field-separator='/' '{print $NF}'`
    URL=`$awk --field-separator='\t' '{print $NF}' "$ADDRESS_FILE" | $sed -e"s/forecast.xml\$/$FORECAST_FILE/"`
    echo -n "  $PLACE: "
    OUTPUT_PATH="$PLACE_DIR/$OUTPUT_FILE"
    $wget --output-document="$OUTPUT_PATH" "$URL" >>"$PLACE_DIR/$LOG_FILE" 2>&1
    if [ "$?" == 0 ]; then 
        echo -n "Ok."
    else
        echo -n "FAILED!"
        rm -f "$OUTPUT_PATH"
        echo "Failed to retrieve forecasts for $PLACE from yr.no" >&2
    fi
    sleep 1
 done
echo ""

echo -n "Retrieving GRIB forecasts for $NOW..."
OUTPUT_FILE="${NOW}_metno-neurope.grb"
OUTPUT_PATH="$GRIB_DIR/$OUTPUT_FILE"
URL="http://api.met.no/weatherapi/gribfiles/1.0/?area=north_europe;content=weather;content_type=application/octet-stream;"
$wget --no-verbose --output-document="$OUTPUT_PATH" "$URL" >>"$GRIB_DIR/$LOG_FILE" 2>&1
if [ "$?" == 0 ]; then 
    echo "Ok."
else
    echo "FAILED!"
    rm -f "$OUTPUT_PATH"
    echo "Failed to retrieve GRIB forecast from yr.no" >&2
fi

