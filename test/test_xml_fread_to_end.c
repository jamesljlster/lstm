#include <stdio.h>
#include <stdlib.h>

#include <lstm.h>
#include <lstm_xml.h>

int main(int argc, char* argv[])
{
	int i, iResult;
	char* tmp;

	for(i = 1; i < argc; i++)
	{
		iResult = lstm_xml_fread_to_end(&tmp, NULL, argv[i]);
		if(iResult < 0)
		{
			printf("Failed to read %s\n", argv[i]);
		}
		else
		{
			printf("%s:\n%s\n", argv[i], tmp);
			free(tmp);
		}
	}

	return 0;
}
