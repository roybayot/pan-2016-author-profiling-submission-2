import xml.etree.ElementTree as ET
import sys
import re

from xml.etree.ElementTree import ParseError

class MyXMLParser(ET.XMLParser):

    rx = re.compile("&#([0-9]+);|&#x([0-9a-fA-F]+);")

    def feed(self,data):
    	mydata = data
        m = self.rx.search(data)
        if m is not None:
            target = m.group(1)
            if target:
                num = int(target)
            else:
                num = int(m.group(2), 16)
            if not(num in (0x9, 0xA, 0xD) or 0x20 <= num <= 0xD7FF
                   or 0xE000 <= num <= 0xFFFD or 0x10000 <= num <= 0x10FFFF):
                # is invalid xml character, cut it out of the stream
                print 'removing %s' % m.group()
                mstart, mend = m.span()
                mydata = data[:mstart] + data[mend:]
        else:
            mydata = data
        super(MyXMLParser,self).feed(mydata)


parser = MyXMLParser(encoding='utf-8')
#xml_filename = sys.argv[1]
xml_filename = "6300ad90bbcee31349ffa0a071ca2041.xml"

try:
    xml_etree = ET.parse(xml_filename, parser=parser)
except ParseError:
    with open(xml_filename, 'r') as f:
        read_data = f.read()
    read_data = read_data.replace("&#11;", "")
    
    with open('temp_file.xml', 'w') as g:
        g.write(read_data)
    xml_etree = ET.parse('temp_file.xml')
except:
    e = sys.exc_info()[0]
    print "Error: %s" % e

