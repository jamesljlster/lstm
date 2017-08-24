#include <stdio.h>
#include <stdlib.h>

#include <lstm.h>
#include <lstm_xml.h>

int main(int argc, char* argv[])
{
	int i, j;
	int ret;

	char** strList;
	int strCount;
	for(i = 1; i < argc; i++)
	{
		printf("Spliting: %s\n", argv[i]);
		ret = lstm_xml_split(&strList, &strCount, argv[i]);
		if(ret != LSTM_NO_ERROR)
		{
			printf("lstm_xml_split() failed!\n");
		}
		else
		{
			for(j = 0; j < strCount; j++)
			{
				printf("%s\n", strList[j]);
			}
			printf("\n");
		}
	}

	return 0;
}
