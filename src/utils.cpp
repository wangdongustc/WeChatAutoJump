#include "utils.h"
#include <glog/logging.h>
#include <iostream>
#include <string>

using namespace std;

string GetCommandOutput(const char *command) {
  FILE *fstream = NULL;
  char buff[400];
  string output;
  fstream = popen(command, "r");
  if (NULL == fstream) {
    LOG(FATAL) << "command " << command << "executed error";
    return NULL;
  }
  while (NULL != fgets(buff, 400, fstream)) {
    output = output.append(buff);
  }
  pclose(fstream);
  return output;
}
