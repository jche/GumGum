# Takes in JSON file with one auction per row and parses it into a data frame
# Data frame has one impression per row and 23 columns for various important variables

# Loading in useful packages
library(dplyr)
library(tidyr)
library(mosaic)
library(tidyjson)
library(readr)
library(lubridate)

text <- read_file("Data/newdata0.txt")
list <- unlist(strsplit(text, "\n"))   # One auction per row

event <- list %>% as.tbl_json %>%
  enter_object("em") %>%
  spread_values(timestamp = jnumber("t"),
                country = jstring("cc"),
                region = jstring("rg"))
event <- event %>%
  mutate(timestamp = round(timestamp, 10)/1000) %>%   # Cut off milliseconds
  mutate(timestamp = as.POSIXct(timestamp, origin="1970-01-01", tz="GMT")) %>%
  mutate(dayofweek = wday(timestamp)) %>%
  mutate(hour = hour(timestamp)) %>%
  mutate(month = month(timestamp)) %>%
  select(-timestamp)

auction <- list %>% as.tbl_json %>%
  enter_object("auction") %>%
  spread_values(margin = jnumber("margin"),
                tmax = jnumber("tmax")) %>%
  enter_object("user") %>%
  spread_values(bluekai = jstring("bkc"))
auction <- auction %>%
  mutate(bluekai = !is.na(bluekai))

site <- list %>% as.tbl_json %>%
  enter_object("auction") %>%
  enter_object("site") %>%
  spread_values(sitetype = jstring("typeid"),
                sitecat = jstring("cat")) %>%
  enter_object("publisher") %>%
  spread_values(pubID = jstring("id"))

bidreq <- list %>% as.tbl_json %>%
  enter_object("auction") %>%
  enter_object("bidrequests") %>%
  gather_array() %>%
  spread_values(reqID1 = jstring("id"),
                vertical = jstring("verticalid"),
                dsp = jstring("bidderid")) %>%
  enter_object("impressions") %>%
  gather_array() %>%
  spread_values(bidfloor = jnumber("bidfloor"),
                adformat = jstring("format"),
                adproduct = jstring("product")) %>%
  enter_object("banner") %>%
  spread_values(width = jnumber("w"),
                height = jnumber("h"))
bidreq <- bidreq %>%
  filter(dsp != "35" & dsp != "37") %>%   # Filter out DSPs we should ignore
  filter(adformat != "1" &   # Filter out Ad Formats we should ignore
           adformat != "2" &
           adformat != "3" &
           adformat != "4" &
           adformat != "6" &
           adformat != "7" &
           adformat != "20" &
           adformat != "21" &
           adformat != "22" & 
           adformat != "25" & 
           adformat != "26")

bid <- list %>% as.tbl_json %>%
  enter_object("auction") %>%
  enter_object("bids") %>%
  gather_array %>%
  spread_values(reqID2 = jstring("requestid"),
                winner = jstring("winner"))

full <- bidreq %>%
  left_join(auction, by="document.id") %>%
  left_join(event, by="document.id") %>%
  left_join(site, by="document.id") %>%
  left_join(bid, by="document.id")
full <- as.data.frame(full)
full <- full %>%   # Checking for bid request response and successful purchase
  mutate(responded = reqID1==reqID2) %>%
  mutate(realwinner = responded & winner == "TRUE") %>%
  select(-winner)
full[["realwinner"]][is.na(full[["realwinner"]])] <- 0
full[["responded"]][is.na(full[["responded"]])] <- 0
full <- full %>%
  select(-reqID1) %>%
  select(-reqID2)
full <- full %>%
  mutate(responded = responded == 1) %>%
  mutate(realwinner = realwinner == 1)


# Makes dummy variables for better ML training
# Breaks with too many observations
# library(dummies)
# spread <- full %>%
#   dummy.data.frame()

# Export df to csv
# write.csv(spread, file="3_15_processed")
