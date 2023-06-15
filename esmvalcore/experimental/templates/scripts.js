function openTab(evt) {
    // Get all elements with class="tablinks" and remove the class "activeTablink"
    var tablinks = document.getElementsByClassName("tablinks");
    for (var i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" activeTablink", "");
    }
    // Add an "activeTablink" class to the button that opened the tab
    evt.currentTarget.className += " activeTablink";

    updateTabVisibility();
}


function toggleDataFiles(button, data_files_id) {
    var data_files = document.getElementById(data_files_id);
    if (data_files.style.display === "none") {
        data_files.style.display = "block";
        button.innerText = "Hide"
    } else {
        data_files.style.display = "none";
        button.innerText = "Show"
    }
}


function hideDataFiles() {
    /***
     * Hides all data file text.
     * The data files are not hidden by default in case the user
     * has disabled javascript and thus can't show them.
     */

    var i, list_data_files;
    list_data_files = document.getElementsByClassName("data_files");
    for (i = 0; i < list_data_files.length; i++) {
        list_data_files[i].style.display = "none";
    }
}

function updateFilter(){
    var filterTypes = document.querySelectorAll(".filter div");
    
    var filters = Array(filterTypes.length);
    for (let i = 0; i < filterTypes.length; i++){
        let filterType = filterTypes[i];
        let checkboxes = document.querySelectorAll("."+ filterType.className +" input[type='checkbox']");
        filters[i] = getSelectedFilters(checkboxes);
    }

    applyFilter(filters);
    updateTabVisibility();
}

function updateTabVisibility(){
    // Find active tablink
    var activeTablink = document.getElementsByClassName("activeTablink")[0];

    // Get all elements with class="tabcontent" and hide them    
    var tabcontent = document.getElementsByClassName("tabcontent");
    for (let tab of tabcontent) {
        var hideEmpty = document.getElementById("cb_hideEmptyDiagnostics").checked;
        let hideThisTab = false;
        if (hideEmpty){
            let figures = tab.getElementsByClassName("div_figure");
            hideThisTab = true;
            for (let fig of figures){
                if (fig.style.display === ""){ // Default value
                    hideThisTab = false;
                    break;
                }
            }
        }
        if (hideThisTab && tab.id === activeTablink.value){
            // Current tab will be hidden, so tab has to be changed to al
            activeTablink.className = activeTablink.className.replace(" activeTablink", "");
            let tabAll = document.getElementById("tabAll");
            tabAll.className += " activeTablink";
            
            // Reset method and leave
            updateTabVisibility();
            return;
        }

        tab.style.display = (
            !hideThisTab
            && (activeTablink.id === "tabAll" || tab.id === activeTablink.value)
            )? "block" : "none";
    }
}


function getSelectedFilters(checkboxes) {
  var classes = [];

    if (checkboxes && checkboxes.length > 0) {
        for (var i = 0; i < checkboxes.length; i++) {
            var cb = checkboxes[i];

            if (cb.checked) {
                classes.push(cb.getAttribute("rel"));
            }
        }
    }

    return classes;
}


function applyFilter(filters){
    var figures = document.getElementsByClassName("div_figure");
    
    if (!figures || figures.length === 0){
        return;
    }
    
    for (const fig of figures){
        // Show figure by returning to default value
        fig.style.removeProperty("display");
        for (const selection of filters){
            if (selection.length === 0) {
                continue;
            }
            let hide = true;
            for (const element of selection){
                if (fig.classList.contains(element)){
                    hide = false;
                }
            }
            if (hide){
                fig.style.display = "none";
                break; // Once it's hidden, there's no need to check more
            }
        }
    }

}


function resetFilter(){
    var filters = document.querySelectorAll(".filter input[type='checkbox']");
    for (const f of filters){
        f.checked = false;
    }
    updateFilter();
}

// Default actions
hideDataFiles();
resetFilter();