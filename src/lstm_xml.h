#ifndef __LSTM_XML_H__
#define __LSTM_XML_H__

// XML attribute structure
struct LSTM_XML_ATTR
{
	char* name;
	char* content;
};

// XML element structure
struct LSTM_XML_ELEM
{
	char* name;
	char* text;

	struct LSTM_XML_ATTR* attrList;
	int attrLen;

	struct LSTM_XML_ELEM* elemList;
	int elemLen;
};

// XML structure
struct LSTM_XML
{
	struct LSTM_XML_ATTR* header;
	int headLen;

	struct LSTM_XML_ELEM* elemList;
	int elemLen;
};

#ifdef __cplusplus
extern "C" {
#endif

// Public functions
int lstm_xml_parse(struct LSTM_XML* xmlPtr, const char* filePath);
void lstm_xml_delete(struct LSTM_XML* xmlPtr);

// Private functions
int lstm_xml_fread_to_end(char** strPtr, const char* filePath);
void lstm_xml_elem_delete(struct LSTM_XML_ELEM* xmlElemPtr);
void lstm_xml_attr_delete(struct LSTM_XML_ATTR* xmlAttrPtr);

#ifdef __cplusplus
}
#endif

#endif
