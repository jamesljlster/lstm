#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lstm.h>
#include <lstm_xml.h>

int main(int argc, char* argv[])
{
	int iResult;

	struct LSTM_XML xml;

	memset(&xml, 0, sizeof(struct LSTM_XML));

	if(argc < 2)
	{
		printf("Assign a xml to run the test\n");
		return 0;
	}

	while(1)
	{
		iResult = lstm_xml_parse(&xml, argv[1]);
		if(iResult != LSTM_NO_ERROR)
		{
			printf("lstm_xml_parse() failed!\n");
		}
		else
		{
			lstm_xml_delete(&xml);
		}
	}

	return 0;
}
