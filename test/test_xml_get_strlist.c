#include <stdio.h>
#include <stdlib.h>

#include <lstm.h>
#include <lstm_xml.h>

int main(int argc, char* argv[])
{
	int i, j, iResult;

	int xmlLen;
	char* xml;

	char** strList;
	int strCount;

	for(i = 1; i < argc; i++)
	{
		iResult = lstm_xml_fread_to_end(&xml, &xmlLen, argv[i]);
		if(iResult < 0)
		{
			printf("Failed to read %s\n", argv[i]);
		}
		else
		{
			iResult = lstm_xml_get_strlist(&strList, &strCount, xml, xmlLen);
			if(iResult != LSTM_NO_ERROR)
			{
				printf("lstm_xml_get_strlist() failed!");
			}
			else
			{
				printf("%s:\n", argv[1]);
				for(j = 0; j < strCount; j++)
				{
					printf("%s\n", strList[j]);
				}
			}
		}
	}

	return 0;
}
