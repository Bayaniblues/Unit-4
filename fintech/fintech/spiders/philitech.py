import scrapy
from ..items import FintechItem


class PhilitechSpider(scrapy.Spider):
    name = 'philitech'
    allowed_domains = ['https://entreplounge.com/100-of-the-best-startups-in-the-philippines/']
    start_urls = ['https://entreplounge.com/100-of-the-best-startups-in-the-philippines/']

    def parse(self, response):
        # call items
        items = FintechItem()

        # extract from css
        # name = response.css('p+ h1').extract()
        description = response.css("p:nth-child(28) , p:nth-child(26) , h1~ p+ p").extract()

        # store our results
        # items["Name"] = name
        items['description'] = description

        yield items
