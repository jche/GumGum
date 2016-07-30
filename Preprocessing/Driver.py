import Event
import Auction
import Auction_Site
import Auction_BidRequests


def process(entry, result, mode="Num"):
    Event.process(entry, result)
    margin = Auction.process(entry, result)
    Auction_Site.process(entry, result)
    Auction_BidRequests.process(margin, entry, result, mode)
    # Response
    result.append(entry["response"])
