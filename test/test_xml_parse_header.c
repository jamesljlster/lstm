#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lstm.h>
#include <lstm_xml.h>

int main(int argc, char* argv[])
{
	int i, j, iResult;
	int indexTmp;
	int fLen;
	char* tmp;

	struct LSTM_XML xml;

	memset(&xml, 0, sizeof(struct LSTM_XML));

	for(i = 1; i < argc; i++)
	{
		iResult = lstm_xml_fread_to_end(&tmp, &fLen, argv[i]);
		if(iResult < 0)
		{
			printf("Failed to read %s\n", argv[i]);
		}

		iResult = lstm_xml_parse_header(&xml, tmp, fLen, &indexTmp);
		if(iResult != LSTM_NO_ERROR)
		{
			printf("lstm_xml_parse_header() failed!\n");
			return -1;
		}
		else
		{
			printf("XML Header:\n");
			for(j = 0; j < xml.headLen; j++)
			{
				printf("%s = %s\n", xml.header[j].name, xml.header[j].content);
			}
		}

		free(tmp);
	}

	return 0;
}
