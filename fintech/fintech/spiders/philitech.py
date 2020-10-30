import scrapy
from ..items import FintechItem


class PhilitechSpider(scrapy.Spider):
    name = 'philitech'
    allowed_domains = ['https://fintechnews.ph/fintech-startups-philippines/']
    start_urls = ['https://fintechnews.ph/fintech-startups-philippines/']

    def parse(self, response):
        # call items
        items = FintechItem()

        # extract from css
        name = response.css('p+ h1').extract()
        description = response.css(".column6:nth-child(1, 2)").extract()

        # store our results
        items["Name"] = name
        items['description'] = description

        yield items
