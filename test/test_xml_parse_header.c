#include <stdio.h>
#include <stdlib.h>

#include <lstm.h>
#include <lstm_xml.h>

int main(int argc, char* argv[])
{
	int i, iResult;
	int indexTmp;
	int fLen;
	char* tmp;

	for(i = 1; i < argc; i++)
	{
		iResult = lstm_xml_fread_to_end(&tmp, &fLen, argv[i]);
		if(iResult < 0)
		{
			printf("Failed to read %s\n", argv[i]);
		}
		else
		{
			printf("%s:\n%s\n", argv[i], tmp);
		}

		iResult = lstm_xml_parse_header(NULL, tmp, fLen, &indexTmp);
		if(iResult != LSTM_NO_ERROR)
		{
			printf("lstm_xml_parse_header() failed!\n");
			return -1;
		}

		free(tmp);
	}

	return 0;
}
