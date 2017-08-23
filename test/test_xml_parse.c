#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lstm.h>
#include <lstm_xml.h>

int main(int argc, char* argv[])
{
	int i, j, iResult;
	int indexTmp;

	struct LSTM_XML xml;

	memset(&xml, 0, sizeof(struct LSTM_XML));

	for(i = 1; i < argc; i++)
	{
		printf("Reading %s\n", argv[i]);
		iResult = lstm_xml_parse(&xml, argv[i]);
		if(iResult != LSTM_NO_ERROR)
		{
			printf("lstm_xml_parse() failed!\n");
		}
		else
		{
			printf("=== Header ===\n");
			for(j = 0; j < xml.headLen; j++)
			{
				printf("%s = %s\n", xml.header[j].name, xml.header[j].content);
			}
			printf("\n");

			lstm_xml_delete(&xml);
		}
	}

	return 0;
}
