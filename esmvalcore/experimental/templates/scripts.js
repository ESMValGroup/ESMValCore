function filterFigures(){
    /**
     * Update visibility of filtered figures.
     */
    let allFigures = $(".div_figure");
    let selectedFigures = allFigures;
    $(".filter_category").each(function() {
        let selection = $(this).find(":checked").map(function() {
            // Returns the figures that the checkbox relates to.
            return $("."+$(this).attr("rel")).get();
        });
        if (selection.length !== 0){
            selectedFigures = selectedFigures.filter(selection);
        }
    });
    selectedFigures.addClass("selected") // affects the div
    .find("figure").show(); // affects figure inside the div
    allFigures.not(selectedFigures).removeClass("selected") // affects the div
    .find("figure").hide(); // affects figure inside the div
}

function filterTabs(){
    /**
     * Disable tab buttons for empty diagnostics and
     * mark empty tabPanes.
     */
    $(".diagnostics-tab").not("#tabAll").each(function() {
        let tabPane = $($(this).attr("data-bs-target"));
        if (tabPane.find(".div_figure.selected").length === 0){
            $(this).addClass("disabled");
            tabPane.addClass("filtered");
        } else {
            $(this).removeClass("disabled");
            tabPane.removeClass("filtered");
        }

        // If the active tab is disabled, change to "All"
        if($(".diagnostics-tab.active").hasClass("disabled")){
            $("#tabAll").click();
        }
    });
}

function hideEmptyTabPanes(){
    /**
     * Hide empty tab panes. It's separated from "filterTabs()"
     * to reuse on the "Hide empty diagnostics" checkbox
     */
    if($("#tabAll").hasClass("active")){
        let panes = $(".diagnostics-tab-pane");
        panes.addClass("active").addClass("show");
        if ($("#cb_hideEmptyDiagnostics").prop("checked")){
            panes.filter(".filtered").removeClass("active").removeClass("show");
        }
    }
}

function applyFilters(){
    /**
     * Updates visibility according to filters.
     */
    filterFigures();
    filterTabs();
    hideEmptyTabPanes();
}

// Set up events with jQuery
// Specific events are defined as anonymous functions
$(document).ready(function() {

    $("#tabAll").on("click", function() {
        /**
         * Functionality for tab "All", as it is not supported
         * by Bootstrap.
         */

        // Both activate this tab
        $(this).addClass("active")
        // and deactivate other tabs
        .parent("li").siblings().find("button").removeClass("active");

        // Show all non-filtered tab panes
        let tabPanes = $(".diagnostics-tab-pane");
        if ($("#cb_hideEmptyDiagnostics").prop("checked")){
            tabPanes = tabPanes.not(".filtered");
        }
        tabPanes.addClass("active").addClass("show");
    });

    $(".diagnostics-tab").not("#tabAll").on("click", function() {
        /**
         * Upgrades Bootstrap tab functionality to deactivate
         * tab "All" by hiding all non-selected panes, as
         * Bootstrap hides only one pane.
         */
        $(".diagnostics-tab-pane").not($(this).attr("data-bs-target"))
        .removeClass("active").removeClass("show");
    });

    // Checkbox "Hide empty diagnostics"
    $("#cb_hideEmptyDiagnostics").on("click", hideEmptyTabPanes);

    $("#b_deleteFilters").on("click", function(){
        /**
         * Unchecks all filters and disables "Delete filters" button.
         */
        $(".filter_cb").prop("checked", false);
        applyFilters();
        $(this).prop("disabled", true);
    });

    $(".filter_cb").on("click", function(){
        /**
         * Update visibility of figures and panes when filters
         * are applied, and set up disable filters button.
         */
        applyFilters();

        let areFiltersClear = $(".filter_cb:checked").length === 0;
        $("#b_deleteFilters").prop("disabled", areFiltersClear);
    });
});
