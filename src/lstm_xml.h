#ifndef __LSTM_XML_H__
#define __LSTM_XML_H__

#include <stdio.h>

// XML parsing state
struct LSTM_XML_PSTAT
{
	int brStr;	// Status of reading string in bracket
	int aStrA;	// Append string to list after append character
	int aStrB;	// Append string to list before append character
};

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
void lstm_xml_fprint(FILE* fptr, struct LSTM_XML* xmlPtr);

// Private functions
void lstm_xml_fprint_attribute(FILE* fptr, struct LSTM_XML_ATTR* attrPtr, int attrLen);
void lstm_xml_fprint_element(FILE* fptr, struct LSTM_XML_ELEM* elemPtr, int indent);
void lstm_xml_fprint_indent(FILE* fptr, int indent);

int lstm_xml_fread_to_end(char** strPtr, int* lenPtr, const char* filePath);
void lstm_xml_elem_delete(struct LSTM_XML_ELEM* xmlElemPtr);
void lstm_xml_attr_delete(struct LSTM_XML_ATTR* xmlAttrPtr);

int lstm_xml_parse_header(struct LSTM_XML* xmlPtr, const char** strList, char*** endPtr);
int lstm_xml_parse_element(struct LSTM_XML* xmlPtr, const char** strList);
int lstm_xml_parse_attribute(char** tagPtr, struct LSTM_XML_ATTR** attrListPtr, int* attrLenPtr, const char* attrStr);

int lstm_xml_get_strlist(char*** strListPtr, const char* xmlSrc, int xmlLen);
int lstm_xml_split(char*** strListPtr, int* strCountPtr, const char* src);

#ifdef __cplusplus
}
#endif

#endif
