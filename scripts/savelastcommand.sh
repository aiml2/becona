#!/bin/bash
echo '#!/bin/sh' >> lastcommmand.sh
 history | tail -n 2 | head -1 | cut -d' ' -f3- >> lastcommmand.sh
