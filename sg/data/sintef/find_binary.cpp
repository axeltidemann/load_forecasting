/********************************************************************
 *   		find_binary.cpp
 *   Created on Tue Feb 07 2012 by Boye A. Hoeverstad.
 *   
 *   Given a list of files, as arguments on the command line and/or as input to
 *   stdin, classify the files as binary or text.
 *******************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <iomanip>
#include <iterator>
#include <unistd.h>

using namespace std;

string program_name;

bool verbose = false;
vector<string> files;
const int chunk_size = 1024*1024*500;
vector<char> buffer(chunk_size);
set<unsigned char> characters;

void
setup_text_character_set()
{
   characters.insert(0x0a); // LF
   characters.insert(0x0d); // CR
   characters.insert(0xe6); // ae
   characters.insert(0xf8); // oe
   characters.insert(0xe5); // aa
   characters.insert(0xc6); // AE
   characters.insert(0xd8); // OE
   characters.insert(0xc5); // AA
}

bool
is_binary(istream &stream, string path)
{
   stream.read(&buffer[0], chunk_size);
   int num_read = stream.gcount();
   for (int n = 0; n < num_read; n++)
   {
      unsigned char c = static_cast<unsigned char>(buffer[n]);
      if ((c < 32 || c > 127) && characters.find(c) == characters.end())
      {
         if (verbose)
         {
            cout << "Binary character: " << ios::hex << static_cast<unsigned int>(c)
                 << " at position " << n << " (probably) of file " 
                 << path << ". Context:\n";
            copy(&buffer[max(n-10, 0)], &buffer[min(n+10, num_read)], ostream_iterator<char>(cout, ""));
            cout << "\n" << flush;
         }
         return true;
      }
   }
   return false;
}

void 
exit_with_usage()
{
   cerr << "Usage: " << program_name << " inputfile [more inputfiles]\n"
        << "Get input files from command line and/or standard input. "
        << "Output an indication of which files are binary and which are text.\n";
   exit(1);
}

void
parse_cmdline_arguments(int argc, char *argv[])
{
   if (argc == 2 && (!strcmp("-?", argv[1]) || !strcmp("--help", argv[1])))
      exit_with_usage();
   int next_arg = 1;
   if (argc >= 2 && (!strcmp("-v", argv[1])))
   {
      next_arg++;
      verbose = true;
   }
   for (; next_arg < argc; next_arg++)
      files.push_back(argv[next_arg]);
}

void
get_stdin_arguments()
{
   if (isatty(fileno(stdin)))
      return;
   string path;
   while (getline(cin, path))
      files.push_back(path);
}

void 
get_input_files(int argc, char *argv[])
{
   parse_cmdline_arguments(argc, argv);
   get_stdin_arguments();
   if (files.size() == 0)
      exit_with_usage();
}

int
main(int argc, char *argv[])
{
   program_name = argv[0];
   setup_text_character_set();
   get_input_files(argc, argv);

   set<string> binary_files, text_files;

   for (vector<string>::const_iterator fit = files.begin(); fit != files.end(); fit++)
   {
      ifstream file(fit->c_str(), ios::binary);
      if (!file)
      {
         cerr << "Failed to open " << *fit << "!\n";
         return 1;
      }
      cout << "Checking file " << *fit << "...\n" << flush;
      bool binary = false;
      while (!file.eof())
         if (is_binary(file, *fit))
            binary = true;
      if (binary)
         binary_files.insert(*fit);
      else
         text_files.insert(*fit);
   }

   cout << "Done.\n\nText files:\n";
   copy(text_files.begin(), text_files.end(), ostream_iterator<string>(cout, "\n"));
   cout << "\nBinary files:\n";
   copy(binary_files.begin(), binary_files.end(), ostream_iterator<string>(cout, "\n"));
   return 0;
}
