/********************************************************************
 *   		anonymize_gs2.cpp
 *   Created on Fri Feb 03 2012 by Boye A. Hoeverstad.
 *   
 *   Anonymizes a series of load measurement files in GS2 (SINTEF Energy
 *   Research / Powel) file format, by replacing the EIA with a random
 *   number. A given EIA will get the same random number for all files given on
 *   the command line (in other words, process all available GS2 files at
 *   once). This is an irreversible process, replacement happens in-place.
 *******************************************************************/

#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <set>
#include <stdexcept>
#include <unistd.h>

using namespace std;

string program_name;
vector<string> files;
set<string> tags;
const streamsize chunk_size = 1024*1024*500;
vector<char> buffer(chunk_size);
map<string, string> replacements;

size_t
find_longest_tag()
{
   size_t longest_tag = 0;
   for (set<string>::const_iterator refit = tags.begin(); 
        refit != tags.end(); refit++)
      longest_tag = max(longest_tag, refit->size());
   return longest_tag;
}

void
setup_tags()
{
   tags.insert("#Reference=");
   tags.insert("#Installation=");
   tags.insert("#Plant=");
}

void 
exit_with_usage()
{
   cerr << "Usage: " << program_name << " inputfile [more inputfiles]\n"
        << "Get input files from command line and/or standard input. "
        << "Search all input files for lines beginning with "
        << "one of the tags (";
   copy(tags.begin(), tags.end(), ostream_iterator<string>(cerr, ", "));
   cerr << "), and replace the rest of the line with a unique random number, "
        << "such that the same EIA always results in the same "
        << "unique random number (across all input files).\n";
   exit(1);
}

void
parse_cmdline_arguments(int argc, char *argv[])
{
   if (argc == 2 && (!strcmp("-?", argv[1]) || !strcmp("--help", argv[1])))
      exit_with_usage();
   for (int arg = 1; arg < argc; arg++)
      files.push_back(argv[arg]);
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

streamsize
read_file(istream &input)
{
   input.read(&buffer[0], chunk_size);
   input.clear();
   streamsize num_read = input.gcount();
   if (num_read == chunk_size)
      throw runtime_error("File too large, doesn't fit in buffer!");
   return num_read;
}

bool
is_tag(const string &tag, const char *input, const size_t tag_len)
{
   return !memcmp(tag.data(), input, tag_len);
}

string
find_EIA(const char *input)
{
   size_t last_digit = 0;
   while (isdigit(input[last_digit]))
      last_digit++;
   return string(input, last_digit);
}

string
create_anonymous_EIA(const string &EIA)
{
   stringstream repl_line;
   repl_line << left << setw(EIA.size()) << rand();
   string anonymous = repl_line.str();
       // Some tags/IDs are short (not really EIAs), in this case select the
       // least significant digits of the random number.
   if (anonymous.size() > EIA.size())
      anonymous = anonymous.substr(anonymous.size() - EIA.size(), EIA.size());
   return anonymous;
}

string
next_anonymous_EIA(const string &EIA)
{
   static set<string> existing_anonymous_EIAs;
   string candidate = create_anonymous_EIA(EIA);
   while (existing_anonymous_EIAs.find(candidate) !=
          existing_anonymous_EIAs.end())
   {
      candidate = create_anonymous_EIA(EIA);
   }
   existing_anonymous_EIAs.insert(candidate);
   return candidate;
};

void
replace_EIA(ostream &output, const char *input, streamsize pos)
{
   string EIA = find_EIA(input);
   map<string, string>::const_iterator eia_it = replacements.find(EIA);
   if (eia_it == replacements.end())
   {
      const string anonymous = next_anonymous_EIA(EIA);
      pair<map<string, string>::iterator, bool> new_pos = 
         replacements.insert(pair<string, string>(EIA, anonymous));
      eia_it = new_pos.first;
   }      
   string anonymous = eia_it->second;
   output.seekp(pos);
   output.write(anonymous.c_str(), anonymous.size());
}

void
anonymize_file(const string &path)
{
   fstream file(path.c_str(), ios::binary | ios::in | ios::out);
   if (!file)
      throw invalid_argument("Failed to open file " + path + "!");

   streamsize last_char = read_file(file) - find_longest_tag();

   int tags_replaced = 0;
   for (set<string>::const_iterator tag_it = tags.begin(); 
        tag_it != tags.end(); tag_it++)
   {
      size_t tag_len = tag_it->size();
      for (streamsize pos = 0; pos < last_char; pos++)
      {
         if (is_tag(*tag_it, &buffer[pos], tag_len))
         {
            pos += tag_it->size();
            replace_EIA(file, &buffer[pos], pos);
            tags_replaced++;
         }
      }
   }
}

int
main(int argc, char *argv[])
{
   try
   {
      program_name = argv[0];
      setup_tags();
      get_input_files(argc, argv);
      for (vector<string>::const_iterator fit = files.begin(); fit != files.end(); fit++)
      {
         cout << "Anonymizing file " << *fit << "..." << flush;
         anonymize_file(*fit);
         cout << " done.\n";
      }
      cout << "All done.\n";
      return 0;
   } 
   catch (exception e)
   {
      cerr << e.what() << endl;
   }
}

