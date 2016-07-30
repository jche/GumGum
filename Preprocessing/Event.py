import Shared as sd
from datetime import datetime


countries_ = ["None", "US", "GB", "CA", "DE", "FR", "NL", "IT"]
regions_ = sd.get_dict("region")


def process(entry, result):
    # Event - time
    t = entry["t"] / 1000

    min = datetime.fromtimestamp(t).minute
    sd.binarize(result, min, 60)

    hour = datetime.fromtimestamp(t).hour
    sd.binarize(result, hour, 24)

    day = datetime.fromtimestamp(t).weekday()
    sd.binarize(result, day, 7)

    hour_of_week = day*24+hour
    sd.binarize(result, hour_of_week, 7*24)

    # Event - country
    sd.add_to_result(result, entry["cc"], countries_)

    # Event - region
    sd.add_to_result(result, entry["rg"], regions_)
