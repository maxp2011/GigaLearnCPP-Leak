#pragma once

#include "Report.h"

void GGL::Report::Display(std::vector<std::string> keyRows) const {
	std::stringstream stream;
	stream << std::string(8, '\n');
	stream << RG_DIVIDER << std::endl;
	std::set<std::string> shownKeys;
	for (std::string row : keyRows) {
		if (!row.empty()) {

			int indentLevel = 0;
			while (row[0] == '-') {
				indentLevel++;
				row.erase(row.begin());
			}

			std::string prefix = {};
			if (indentLevel > 0) {
				prefix += std::string((indentLevel - 1) * 3, ' ');
				prefix += " - ";
			}
			if (Has(row)) {
				stream << prefix << SingleToString(row, true) << std::endl;
				shownKeys.insert(row);
			} else {
				continue;
			}
		} else {
			stream << std::endl;
		}
	}

	// Show any extra metrics (e.g. from StepCallback: Player/*, Game/*, Rewards/*)
	for (const auto& pair : data) {
		if (shownKeys.find(pair.first) == shownKeys.end()) {
			stream << " " << SingleToString(pair.first, true) << std::endl;
		}
	}

	stream << std::string(4, '\n');

	std::cout << stream.str();
}
