#!/bin/bash

_PANDOC_CMD=""
_MD_FILE=""
_NO_LISTINGS=0
_CHAPTER_NUMBERS=0
_PDF_ENGINE="xelatex"

function print_helper_msg(){
cat <<EOF

bash scripts/generate_pdf.sh <options>

-h, --help              # Print this helper message
-f, --file <file_path>  # The file path of the markdown file
-N, --numbers           # Add chapter numbers to headers

EOF
}

function parse_args_from_console() {
    local prompt_str=$1

    while [ "$1" != "--" ] && [[ $# -gt 0 ]]; do
        case $1 in
            -f | --file)                              shift
                                                      _MD_FILE=${1%.*}
                                                      ;;
            -e | --pdf-engine)                        shift
                                                      _PDF_ENGINE=$1
                                                      ;;
            -N | --numbers)                           _CHAPTER_NUMBERS=1
                                                      ;;
            -nl | --nolistings)                       _NO_LISTINGS=1
                                                      ;;
            -h | --help)                              print_helper_msg
                                                      exit 0
                                                      ;;
            *)                                        print_helper_msg
                                                      exit 0
                                                      ;;
        esac
        shift
    done
    shift
    _PANDOC_CMD=$@
    if [[ ${#_PANDOC_CMD} == 0 ]]; then
      _PANDOC_CMD=""
    fi
}


_SCRIPT_PATH=$(readlink -f $0)
WorkSpaceRootDir="$(dirname $(dirname ${_SCRIPT_PATH#"generate_pdf.sh"#}}))"
_TEMPLATE_YAML="${WorkSpaceRootDir}/gen_docs_config/eisvogel.latex"
_TEMPLATE_CLI="--template ${_TEMPLATE_YAML}"

parse_args_from_console $@

_PANDOC_CMDLINE=""
_PANDOC_CMDLINE+="${_MD_FILE}.md -o ${_MD_FILE}.pdf "
_PANDOC_CMDLINE+="--from markdown "
_PANDOC_CMDLINE+="${_TEMPLATE_CLI} "
_PANDOC_CMDLINE+="--pdf-engine=${_PDF_ENGINE} "
_PANDOC_CMDLINE+="--shift-heading-level-by=-1 "


if [[ ${_NO_LISTINGS} -eq 0 ]]; then
  _PANDOC_CMDLINE+="--listings "
fi

if [[ ${_CHAPTER_NUMBERS} -eq 1 ]]; then
  _PANDOC_CMDLINE+="-N "
fi

_PANDOC_CMDLINE+="${_PANDOC_CMD}"

# echo "Pandoc command line: ${_PANDOC_CMDLINE}"
docker run --rm -v "$(pwd):/data" alwaysproblem/pandoc ${_PANDOC_CMDLINE}
# docker run --rm -ti -v "$(pwd):/data" "${_DOCKER_ARGS[@]}" --entrypoint "/bin/bash" alwaysproblem/pandoc
