# coding=utf-8

from nowatermark import WatermarkRemover

path = './data2/'

watermark_template_filename = path + '365.jpg'
remover = WatermarkRemover()
remover.load_watermark_template(watermark_template_filename)

remover.remove_watermark(path + '1.jpg', path + '1-result.jpg')
remover.remove_watermark(path + '2.jpg', path + '2-result.jpg')
remover.remove_watermark(path + '3.jpg', path + '3-result.jpg')
remover.remove_watermark(path + '4.jpg', path + '4-result.jpg')
remover.remove_watermark(path + '5.jpg', path + '5-result.jpg')

# path = './data1/'

# watermark_template_filename = path + 'anjuke-watermark-template.jpg'
# remover = WatermarkRemover()
# remover.load_watermark_template(watermark_template_filename)

# remover.remove_watermark(path + 'anjuke2.jpg', path + 'anjuke2-result.jpg')
# remover.remove_watermark(path + 'anjuke3.jpg', path + 'anjuke3-result.jpg')
# remover.remove_watermark(path + 'anjuke4.jpg', path + 'anjuke4-result.jpg')
# remover.remove_watermark(path + 'anjuke5.jpg', path + 'anjuke5-result.jpg')